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
import pytorch_cifar_models
import timm

from data import data_loaders
from utils import count_parameters, _ECELoss, ConvNet

from laplace import Laplace

parser = argparse.ArgumentParser(description='LA for DNNs')
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
parser.add_argument('--job-id', default='laplace', type=str)

parser.add_argument('--subset-of-weights', default='all', choices=['all', 'subnetwork', 'last_layer'], type=str)
parser.add_argument('--hessian-structure', default='full', choices=['full', 'kron', 'lowrank', 'diag'], type=str)

parser.add_argument('--K', type=int, default=20)

def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.dataset, args.arch, args.job_id)
	args.num_classes = 10 if args.dataset in ['mnist', 'cifar10'] else (100 if args.dataset == 'cifar100' else 1000)

	if args.data_root is None:
		if 'cifar' in args.dataset:
			if os.path.isdir('/data/LargeData/Regular/cifar/'):
				args.data_root = '/data/LargeData/Regular/cifar/'
		elif args.dataset == 'imagenet':
			if os.path.isdir('/data/LargeData/Large/ImageNet/'):
				args.data_root = '/data/LargeData/Large/ImageNet/'
			elif os.path.isdir('/workspace/home/zhijie/ImageNet/'):
				args.data_root = '/workspace/home/zhijie/ImageNet/'
		elif args.dataset == 'mnist':
			if os.path.isdir('/data/LargeData/Regular/'):
				args.data_root = '/data/LargeData/Regular/'
		else:
			assert False

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

	train_loader_noaug, val_loader, test_loader = data_loaders(args,
		valid_size=0.01 if args.dataset == 'imagenet' else 0.2, noaug=True)
	print("Number of training data", len(train_loader_noaug.dataset))

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

	model.eval()
	print("---------MAP model ---------")
	test(test_loader, model, device, args)

	###### LA ######
	print("--------- LA ---------")
	la = Laplace(model, 'classification',
				 subset_of_weights=args.subset_of_weights,
				 hessian_structure=args.hessian_structure)
	la.fit(train_loader_noaug)
	la.optimize_prior_precision(method='marglik', link_approx='mc', n_samples=args.K)
	test(test_loader, la, device, args, laplace=True)

	if 'cifar' in args.dataset:
		print("--------- MAP on corrupted_data ---------")
		eval_corrupted_data(model, device, args, token='map')

		print("--------- LA on corrupted_data ---------")
		eval_corrupted_data(la, device, args, laplace=True, token=args.job_id)


def test(test_loader, model, device, args, laplace=False, verbose=True):
	if not laplace:
		model.eval()

	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x_batch, y_batch in test_loader:
			x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
			if laplace:
				y_pred = model(x_batch, pred_type='glm', link_approx='mc', n_samples=args.K).log()
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
		print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))
	return loss, acc, ece

def eval_corrupted_data(model, device, args, laplace=False, token=''):
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
			test_loss, top1, ece = test(corrupted_loader, model, device, args, laplace=laplace, verbose=False)
			results[i, ii] = np.array([test_loss, top1, ece])
	print(results.mean(1)[:, 2])
	np.save(args.save_dir + '/corrupted_results_{}.npy'.format(token), results)

if __name__ == '__main__':
	main()
