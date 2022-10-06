import time
from collections import OrderedDict
import os
import re
import math
import argparse
import numpy as np
import copy
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as trn
import pytorch_cifar_models
import timm

from data import data_loaders
from utils import count_parameters, _ECELoss, ConvNet

from scalablebdl.mean_field import to_bayesian
from scalablebdl.bnn_utils import use_flipout, use_single_eps
from scalablebdl.prior_reg import PriorRegularizor

parser = argparse.ArgumentParser(description='MFVI for DNNs')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--not-use-gpu', action='store_true', default=False)

parser.add_argument('-b', '--batch-size', default=100, type=int,
					metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--test-batch-size', default=128, type=int)

parser.add_argument('--arch', type=str, default='cifar10_resnet20')
parser.add_argument('--dataset', type=str, default='cifar10',
					choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data-root', type=str, default=None)
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
					help='path to pretrained MAP checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='./logs/', type=str)
parser.add_argument('--job-id', default='mfvi', type=str)

parser.add_argument('--K', default=20, type=int)
parser.add_argument('--epochs', metavar='N', type=int, default=40, help='Number of epochs to ft.')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--ft_lr', type=float, default=8e-4, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.dataset, args.arch, args.job_id)
	args.num_classes = 10 if args.dataset in ['mnist', 'cifar10'] else (100 if args.dataset == 'cifar100' else 1000)

	if args.data_root is None:
		args.data_root = '/data/LargeData/Regular/cifar' if args.dataset == 'cifar10' else '/data/LargeData/Large/ImageNet'
		# assert False

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if not args.not_use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cudnn.benchmark = True
	else:
		device = torch.device('cpu')

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	train_loader, test_loader = data_loaders(args)
	print("Number of training data", len(train_loader.dataset))

	if args.arch == 'mnist_model':
		model = ConvNet(args.num_classes).to(device)
	elif args.arch in pytorch_cifar_models.__dict__:
		model = pytorch_cifar_models.__dict__[args.arch](pretrained=True).to(device)
	elif args.arch in models.__dict__:
		model = models.__dict__[args.arch](pretrained=True).to(device)
	elif 'vit' in args.arch:
		assert args.dataset == 'imagenet'
		model = timm.create_model(args.arch, pretrained=True).to(device)
	elif args.arch == "small_deq":
		from modules.model import ResDEQ
		model = ResDEQ(3, 48, 64, args.num_classes).float().to(device)
	else:
		model = nn.Sequential(nn.Flatten(1), nn.Linear(3072, args.num_classes))

	params = {name: p for name, p in model.named_parameters()}
	print("Number of parameters", count_parameters(model))
	if args.pretrained is not None:
		print("Load MAP model from", args.pretrained)
		model.load_state_dict(torch.load(args.pretrained))

	print("---------MAP model ---------")
	test(test_loader, model, device, args)

	print("--------- MFVI ---------")
	mfvi_model = to_bayesian(model.cpu(), num_mc_samples=args.K).to(device)
	pre_trained, new_added = [], []
	for name, param in mfvi_model.named_parameters():
		if 'psi' in name:
			new_added.append(param)
		else:
			pre_trained.append(param)
	args.lr_min = args.ft_lr
	optimizer = torch.optim.SGD([{'params': pre_trained, 'lr': args.ft_lr},
								 {'params': new_added, 'lr': args.lr}],
								 weight_decay=0, momentum=args.momentum)
	prior_reg = PriorRegularizor(mfvi_model, args.decay, len(train_loader.dataset),
										 args.K, 'mean_field', False)

	for epoch in range(args.epochs):
		use_flipout(mfvi_model)
		mfvi_finetune(epoch, train_loader, mfvi_model, optimizer, prior_reg, device, args)
		use_single_eps(mfvi_model)
		test(test_loader, mfvi_model, device, args, is_mfvi=True)

	if args.dataset in ['cifar10', 'cifar100', 'imagenet']:
		print("--------- MFVI on corrupted_data ---------")
		eval_corrupted_data(mfvi_model, device, args, is_mfvi=True, token='mfvi')

	if args.dataset in ['cifar10', 'cifar100']:
		if args.dataset == 'cifar10':
			normalize = trn.Normalize(mean=[0.4914, 0.4822, 0.4465],
										 	 std=[0.2023, 0.1994, 0.201])
		else:
			normalize = trn.Normalize(mean=[0.507, 0.4865, 0.4409],
										 	 std=[0.2673, 0.2564, 0.2761])
		ood_dataset = dset.SVHN(args.data_root.replace('cifar', 'svhn'),
			split='test', download=True,
			transform=trn.Compose([
				trn.ToTensor(),
				normalize,
			]))
		ood_loader = torch.utils.data.DataLoader(ood_dataset,
			batch_size=args.test_batch_size, num_workers=args.workers,
			pin_memory=True, shuffle=False)
		is_correct, confidences = test(test_loader, mfvi_model, device, args, is_mfvi=True, return_more=True)
		is_correct_ood, confidences_ood = test(ood_loader, mfvi_model, device, args, is_mfvi=True, return_more=True)

		is_correct = torch.cat([is_correct, torch.zeros_like(is_correct_ood)]).data.cpu().numpy()
		confidences = torch.cat([confidences, confidences_ood]).data.cpu().numpy()

		ths = np.linspace(0, 1, 300)[:299]
		accs = []
		for th in ths:
			# if th >= confidences.max():
			# 	accs.append(accs[-1])
			# else:
			accs.append(is_correct[confidences > th].mean())
		np.save(args.save_dir + '/acc_vs_conf_{}.npy'.format(args.job_id), np.array(accs))

def mfvi_finetune(epoch, train_loader, model, optimizer, prior_reg, device, args, verbose=True):
	model.train()
	losses, acc, num_data = 0, 0, 0
	for i, (x_batch, y_batch) in enumerate(train_loader):
		cur_lr, cur_slr = adjust_learning_rate_per_step(
					optimizer, epoch, i, len(train_loader), args)
		x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

		y_pred = model(x_batch)
		loss = F.cross_entropy(y_pred, y_batch)

		with torch.no_grad():
			losses += loss.item() * x_batch.shape[0]
			acc += (torch.max(y_pred, 1)[1] == y_batch).float().sum().item()
			num_data += x_batch.shape[0]

		optimizer.zero_grad()
		loss.backward()
		prior_reg.step()
		optimizer.step()

	losses /= num_data
	acc /= num_data
	if verbose:
		print("Epoch {} of MFVI finetuning: loss: {:.4f}, Accuracy: {:.4f}, lr: {:.4f} {:.4f}".format(epoch, losses, acc, cur_lr, cur_slr))

def test(test_loader, model, device, args, is_mfvi=False, verbose=True, return_more=False):
	t0 = time.time()

	model.eval()

	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x_batch, y_batch in test_loader:
			x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

			if is_mfvi:
				y_pred = 0
				for _ in range(args.K):
					y_pred += model(x_batch).softmax(-1)
				y_pred = y_pred.div(args.K).log()
			else:
				y_pred = model(x_batch)

			loss += F.cross_entropy(y_pred, y_batch).item() * x_batch.shape[0]
			conf, pred = torch.max(y_pred.softmax(-1), 1)
			acc += (pred == y_batch).float().sum().item()
			num_data += x_batch.shape[0]

			targets.append(y_batch)
			confidences.append(conf)
			predictions.append(pred)

		targets, confidences, predictions = torch.cat(targets), torch.cat(confidences), torch.cat(predictions)
		loss /= num_data
		acc /= num_data
		ece = _ECELoss()(confidences, predictions, targets).item()

	if verbose:
		print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}, time: {:.2f}s".format(loss, acc, ece, time.time() - t0))
	if return_more:
		return (targets == predictions).float(), confidences
	return loss, acc, ece

def eval_corrupted_data(model, device, args, is_mfvi=False, token=''):
	if 'cifar' in args.dataset:
		if args.dataset == 'cifar10':
			corrupted_data_path = './CIFAR-10-C/CIFAR-10-C'
			mean=torch.tensor([0.4914, 0.4822, 0.4465])
			std=torch.tensor([0.2023, 0.1994, 0.201])
		else:
			corrupted_data_path = './CIFAR-100-C/CIFAR-100-C'
			mean=torch.tensor([0.507, 0.4865, 0.4409])
			std=torch.tensor([0.2673, 0.2564, 0.2761])

		corrupted_data_files = os.listdir(corrupted_data_path)
		corrupted_data_files.remove('labels.npy')
		if 'README.txt' in corrupted_data_files:
			corrupted_data_files.remove('README.txt')
		results = np.zeros((5, len(corrupted_data_files), 3))
		labels = torch.from_numpy(np.load(os.path.join(corrupted_data_path, 'labels.npy'), allow_pickle=True)).long()
		for ii, corrupted_data_file in enumerate(corrupted_data_files):
			corrupted_data = np.load(os.path.join(corrupted_data_path, corrupted_data_file), allow_pickle=True)
			for i in range(5):
				print(corrupted_data_file, i)
				images = torch.from_numpy(corrupted_data[i*10000:(i+1)*10000]).float().permute(0, 3, 1, 2)/255.
				images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
				corrupted_dataset = torch.utils.data.TensorDataset(images, labels[i*10000:(i+1)*10000])
				corrupted_loader = torch.utils.data.DataLoader(corrupted_dataset,
					batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
					pin_memory=False, sampler=None, drop_last=False)
				test_loss, top1, ece = test(corrupted_loader, model, device, args, is_mfvi=is_mfvi, verbose=False)
				results[i, ii] = np.array([test_loss, top1, ece])
	elif args.dataset == 'imagenet':
		mean=torch.tensor([0.485, 0.456, 0.406])
		std=torch.tensor([0.229, 0.224, 0.225])

		distortions = [
			'gaussian_noise', 'shot_noise', 'impulse_noise',
			'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
			'snow', 'frost', 'fog', 'brightness',
			'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
			'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
		]

		results = np.zeros((5, len(distortions), 3))
		for ii, distortion_name in enumerate(distortions):
			print(distortion_name)
			for i, severity in enumerate(range(1, 6)):
				corrupted_dataset = dset.ImageFolder(
					root='../imagenet-c/' + distortion_name + '/' + str(severity),
					transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

				corrupted_loader = torch.utils.data.DataLoader(
					corrupted_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

				test_loss, top1, ece = test(corrupted_loader, model, device, args, is_mfvi=is_mfvi, verbose=False)
				results[i, ii] = np.array([test_loss, top1, ece])
	else:
		raise NotImplementedError
	print(results.mean(1)[:, 2])
	np.save(args.save_dir + '/corrupted_results_{}.npy'.format(token), results)

def adjust_learning_rate_per_step(optimizer, epoch, i, num_ites_per_epoch, args):
	optimizer.param_groups[0]['lr'] = args.ft_lr# * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
	optimizer.param_groups[1]['lr'] = args.lr_min + (args.lr-args.lr_min) * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
	return optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']

if __name__ == '__main__':
	main()
