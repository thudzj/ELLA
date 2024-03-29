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

from data import data_loaders, subsample
from utils import count_parameters, _ECELoss, build_dual_params_list, jac, \
	Psi_raw, psd_safe_cholesky, ConvNet, check_approx_error, measure_speed

parser = argparse.ArgumentParser(description='LLA for DNNs')
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
parser.add_argument('--job-id', default='default', type=str)
parser.add_argument('--resume-dual-params', default=None, type=str)
parser.add_argument('--resume-cov-inv', default=None, type=str)

parser.add_argument('--K', default=20, type=int)
parser.add_argument('--M', default=100, type=int, help='the number of samples')
parser.add_argument('--I', default=1, type=int, help='the number of samples')
parser.add_argument('--balanced', action='store_true', default=False)
parser.add_argument('--not-random', action='store_true', default=False)

# parser.add_argument('--early-stop', default=None, type=int)
parser.add_argument('--search-freq', default=None, type=int)
parser.add_argument('--sigma2', default=0.1, type=float)

parser.add_argument('--num-samples-eval', default=512, type=int,metavar='N')
parser.add_argument('--ntk-std-scale', default=1, type=float)

parser.add_argument('--check', action='store_true', default=False)
parser.add_argument('--measure-speed', action='store_true', default=False)
parser.add_argument('--track-test-results', action='store_true', default=False)


def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.dataset, args.arch, args.job_id)
	args.num_classes = 10 if args.dataset in ['mnist', 'cifar10'] else (100 if args.dataset == 'cifar100' else 1000)
	args.random = not args.not_random

	if args.M < args.num_classes:
		args.balanced = False

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

	model_bk = copy.deepcopy(model)

	###### ella ######
	## check the approximation error
	if args.check:
		check_approx_error(args, model, params, train_loader_noaug, val_loader, device)

	if args.resume_dual_params is not None:
		if args.resume_dual_params == 'auto':
			args.resume_dual_params = os.path.join(args.save_dir, 'dual_params.tar.gz')
		ckpt = torch.load(args.resume_dual_params)
		dual_params_list = [{k:v.to(device) for k,v in dual_params.items()} for dual_params in ckpt['0']]
	else:
		x_subsample, y_subsample = subsample(train_loader_noaug, args.num_classes,
											 args.M, args.balanced,
											 device, verbose=False)
		dual_params_list = build_dual_params_list(model, params, x_subsample, y_subsample, args=args, num_batches=args.I)

		torch.save({
			'0': [{k:v.data.cpu() for k,v in dual_params.items()} for dual_params in dual_params_list],
		}, os.path.join(args.save_dir if not 'vit' in args.arch else '../ella_logs', 'dual_params.tar.gz'))

	if args.measure_speed:
		x, y = subsample(train_loader_noaug, args.num_classes, args.batch_size, False, device, verbose=False)
		measure_speed(model, params, dual_params_list, model_bk, x, y)

	Psi = partial(Psi_raw, model, params, dual_params_list)

	if args.resume_cov_inv is not None:
		if args.resume_cov_inv == 'auto':
			args.resume_cov_inv = os.path.join(args.save_dir, 'cov_inv.tar.gz')
		ckpt = torch.load(args.resume_cov_inv)
		best_cov_inv = ckpt['0'].to(device)
	else:
		## pass the training set
		best_value = 1e8; best_cov_inv = None; best_test_results = None
		test_results_list = []
		with torch.no_grad():
			cov = torch.zeros(args.K, args.K).cuda(non_blocking=True)
			for i, (x, y) in enumerate(train_loader_noaug):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
				Psi_x, logits = Psi(x, return_output=True)
				prob = logits.softmax(-1)
				Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
				cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x, Psi_x)

				if args.search_freq and (i + 1) * args.batch_size % args.search_freq == 0:
					cov_clone = cov.data.clone()
					cov_clone.diagonal().add_(1 / args.sigma2 * (i + 1) * args.batch_size / len(train_loader_noaug.dataset))
					cov_inv = cov_clone.inverse()
					val_loss, _, _ = ella_test(val_loader, model, device, args, Psi, cov_inv, verbose=True)
					if args.track_test_results:
						test_loss, test_acc, test_ece = ella_test(test_loader, model, device, args, Psi, cov_inv, verbose=False)
						test_results_list.append(np.array([(i + 1) * args.batch_size, test_loss, test_acc, test_ece]))
					if val_loss < best_value:
						best_value = val_loss
						best_cov_inv = cov_inv
						test_loss, test_acc, test_ece = ella_test(test_loader, model, device, args, Psi, best_cov_inv, verbose=False)
						best_test_results = "Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(test_loss, test_acc, test_ece)

						torch.save({
							'0': best_cov_inv.data.cpu(),
						}, os.path.join(args.save_dir, 'cov_inv.tar.gz'))

					print("Current training data {}, loss: {:.4f}, best loss: {:.4f}"
						  "\n    {}".format((i + 1) * args.batch_size, val_loss, best_value, best_test_results))

		if args.track_test_results:
			test_results_list = np.stack(test_results_list)
			np.save(args.save_dir + '/test_results.npy', test_results_list)

	print("--------- LLA ---------")
	ella_test(test_loader, model, device, args, Psi, best_cov_inv)

	if args.dataset in ['cifar10', 'cifar100', 'imagenet']:
		print("--------- MAP on corrupted_data ---------")
		eval_corrupted_data(model_bk, device, args, token='map')

		print("--------- LLA on corrupted_data ---------")
		eval_corrupted_data(model, device, args, Psi, best_cov_inv, token='ella')

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

		is_correct, confidences = ella_test(test_loader, model, device, args, Psi, best_cov_inv, return_more=True)
		is_correct_ood, confidences_ood = ella_test(ood_loader, model, device, args, Psi, best_cov_inv, return_more=True)

		is_correct = torch.cat([is_correct, torch.zeros_like(is_correct_ood)]).data.cpu().numpy()
		confidences = torch.cat([confidences, confidences_ood]).data.cpu().numpy()

		ths = np.linspace(0, 1, 300)[:299]
		accs = []
		for th in ths:
			# if th >= confidences.max():
			# 	accs.append(accs[-1])
			# else:
				accs.append(is_correct[confidences > th].mean())
		np.save(args.save_dir + '/acc_vs_conf_ella.npy', np.array(accs))

		is_correct, confidences = test(test_loader, model_bk, device, args, return_more=True)
		is_correct_ood, confidences_ood = test(ood_loader, model_bk, device, args, return_more=True)

		is_correct = torch.cat([is_correct, torch.zeros_like(is_correct_ood)]).data.cpu().numpy()
		confidences = torch.cat([confidences, confidences_ood]).data.cpu().numpy()

		ths = np.linspace(0, 1, 300)[:299]
		accs = []
		for th in ths:
			# if th >= confidences.max():
			# 	accs.append(accs[-1])
			# else:
				accs.append(is_correct[confidences > th].mean())
		np.save(args.save_dir + '/acc_vs_conf_map.npy', np.array(accs))

def ella_test(test_loader, model, device, args, Psi, cov_inv, verbose=True, return_more=False):
	t0 = time.time()
	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			Psi_x, y_pred = Psi(x, return_output=True)
			F_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)
			# if 'vit' in args.arch:
			# 	F_var_L = torch.stack([psd_safe_cholesky(item) for item in F_var])
			# else:
			F_var_L = psd_safe_cholesky(F_var)
			F_samples = (F_var_L @ torch.randn(F_var.shape[0], F_var.shape[1], args.num_samples_eval,
				device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + y_pred
			prob = F_samples.softmax(-1).mean(0)

			loss += F.cross_entropy(prob.log(), y).item() * x.shape[0]
			conf, pred = torch.max(prob, 1)
			acc += (pred == y).float().sum().item()
			num_data += x.shape[0]

			targets.append(y)
			confidences.append(conf)
			predictions.append(pred)

		targets, confidences, predictions = torch.cat(targets), torch.cat(confidences), torch.cat(predictions)
		loss /= num_data
		acc /= num_data
		ece = _ECELoss()(confidences, predictions, targets).item()
	if verbose:
		print("Test results of ELLA: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}, time: {:.2f}s".format(loss, acc, ece, time.time() - t0))
	if return_more:
		return (targets == predictions).float(), confidences
	return loss, acc, ece

def test(test_loader, model, device, args, verbose=True, return_more=False):
	t0 = time.time()

	model.eval()

	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x_batch, y_batch in test_loader:
			x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
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

def eval_corrupted_data(model, device, args, Psi=None, cov_inv=None, token=''):
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
				if Psi is None:
					test_loss, top1, ece = test(corrupted_loader, model, device, args, verbose=False)
				else:
					test_loss, top1, ece = ella_test(corrupted_loader, model, device, args, Psi, cov_inv, verbose=False)
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

				if Psi is None:
					test_loss, top1, ece = test(corrupted_loader, model, device, args, verbose=False)
				else:
					test_loss, top1, ece = ella_test(corrupted_loader, model, device, args, Psi, cov_inv, verbose=False)
				results[i, ii] = np.array([test_loss, top1, ece])
	else:
		raise NotImplementedError
	print(results.mean(1)[:, 2])
	np.save(args.save_dir + '/corrupted_results_{}.npy'.format(token), results)

if __name__ == '__main__':
	main()
