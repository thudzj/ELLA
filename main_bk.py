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
import torch.autograd.forward_ad as fwAD

import torchvision.models as models
import pytorch_cifar_models

from data import data_loaders, nystrom_subsample
from utils import count_parameters, _ECELoss, psd_safe_eigen, psd_safe_cholesky, find_module_by_name

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
					choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--data-root', type=str, default=None)
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
					help='path to pretrained MAP checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='./logs/', type=str)
parser.add_argument('--job-id', default='default', type=str)

parser.add_argument('--nystrom-samples', default=100, type=int,
					metavar='N', help='subsample size for nystrom')
parser.add_argument('--balanced-sampling', action='store_true', default=False)
parser.add_argument('--use-normalizer', action='store_true', default=False)

parser.add_argument('--early-stop', default=None, type=int)
parser.add_argument('--search-freq', default=None, type=int)
parser.add_argument('--sigma2', default=0.1, type=float)

parser.add_argument('--num-samples-eval', default=256, type=int,
					metavar='N', help='subsample size for nystrom')
parser.add_argument('--ntk-std-scale', default=1, type=float)

def main():
	args = parser.parse_args()
	args.save_dir = os.path.join(args.save_dir, args.job_id)
	args.num_classes = 10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 1000)

	if args.data_root is None:
		if 'cifar' in args.dataset:
			if os.path.isdir('/data/LargeData/Regular/cifar/'):
				args.data_root = '/data/LargeData/Regular/cifar/'
		elif args.dataset == 'imagenet':
			if os.path.isdir('/data/LargeData/Large/ImageNet/'):
				args.data_root = '/data/LargeData/Large/ImageNet/'
		else:
			assert False

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	device = torch.device('cuda') if not args.not_use_gpu and torch.cuda.is_available() else torch.device('cpu')

	train_loader_noaug, val_loader, test_loader = data_loaders(args,
		valid_size=0.2 if 'cifar' in args.dataset else 0.01, noaug=True)

	if args.arch in pytorch_cifar_models.__dict__:
		model = pytorch_cifar_models.__dict__[args.arch](pretrained=True).to(device)
	elif args.arch in models.__dict__:
		model = models.__dict__[args.arch](pretrained=True).to(device)
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

	###### lla ######

	## calculate eigenfunctions by the nystrom method
	x_nystrom, y_nystrom = nystrom_subsample(train_loader_noaug, args)
	# print(np.unique(y_nystrom.numpy(), return_counts=True))
	x_nystrom, y_nystrom = x_nystrom.to(device), y_nystrom.to(device)
	g_nystrom = batch_grad(model, x_nystrom, y_nystrom, args)
	K = torch.einsum('np,mp->nm', g_nystrom, g_nystrom).to(device)

	normalizer = K.diagonal().mean() if args.use_normalizer else 1
	K = K / normalizer
	p, q = psd_safe_eigen(K)
	eigenvalues = p / args.nystrom_samples
	Psi = partial(Psi_raw, model, params, g_nystrom, args, p, q, normalizer)
	eigenfuncs = partial(eigenfuncs_raw, model, params, g_nystrom, args, p, q, normalizer)

	## check if the nystrom method is correct
	print(eigenfuncs(x_nystrom, y_nystrom) @ torch.diag(eigenvalues) @ eigenfuncs(x_nystrom, y_nystrom).T)
	print(K)
	print(torch.dist(eigenfuncs(x_nystrom, y_nystrom) @ torch.diag(eigenvalues) @ eigenfuncs(x_nystrom, y_nystrom).T, K), torch.dist(K, torch.zeros_like(K)))

	## pass the training set
	best_value = 1e8; best = None; best_test_results = None
	with torch.no_grad():
		cov = torch.zeros(args.nystrom_samples, args.nystrom_samples).cuda(non_blocking=True)
		for i, (x, y) in tqdm(enumerate(train_loader_noaug), desc='Passing the training set', total=len(train_loader_noaug)):
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			Psi_x, logits = Psi(x, y=y, return_logits=True)
			prob = logits.softmax(-1)
			Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
			cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x, Psi_x)

			if args.search_freq and (i + 1) * args.batch_size % args.search_freq == 0:
				cov_clone = cov.data.clone()
				cov_clone.diagonal().add_(1/args.sigma2/normalizer)
				cov_inv = cov_clone.inverse()
				val_loss, _, _ = lla_test(val_loader, model, device, args, Psi, cov_inv, verbose=False)
				if val_loss < best_value:
					best_value = val_loss
					best = (i + 1) * args.batch_size
					test_loss, test_acc, test_ece = lla_test(test_loader, model, device, args, Psi, cov_inv, verbose=False)
					best_test_results = "Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(test_loss, test_acc, test_ece)
				print("Current subsample number: {}, loss: {:.4f}, best subsample number: {}, best loss: {:.4f}"
					  "\n    {}".format((i + 1) * args.batch_size, val_loss, best, best_value, best_test_results))

			if args.early_stop and (i + 1) * args.batch_size >= args.early_stop:
				print("--------- LLA model ---------")
				cov.diagonal().add_(1/args.sigma2/normalizer)
				cov_inv = cov.inverse()
				lla_test(test_loader, model, device, args, Psi, cov_inv)
				return

def Psi_raw(model, params, g_nystrom, args, p, q, normalizer, x, y=None, return_logits=False):
	Kbn, logits, a = nystrom_kernel(model, params, x, g_nystrom, args, y, return_more=True)
	if return_logits:
		return a.unsqueeze(-1) @ (Kbn @ q / p.add(1e-8).sqrt() / normalizer).unsqueeze(1), logits
	else:
		return a.unsqueeze(-1) @ (Kbn @ q / p.add(1e-8).sqrt() / normalizer).unsqueeze(1)

def eigenfuncs_raw(model, params, g_nystrom, args, p, q, normalizer, x, y=None):
	Kbn = nystrom_kernel(model, params, x, g_nystrom, args, y)
	return Kbn @ q / p.add(1e-8) / normalizer * math.sqrt(args.nystrom_samples)

@torch.no_grad()
def nystrom_kernel(model, params, x_batch, g_nystrom, args, y_batch=None, return_more=False):

	with fwAD.dual_level():
		Jvs = []
		for g in g_nystrom:
			start = 0
			for name, p in params.items():
				module, name_p = find_module_by_name(model, name)
				delattr(module, name_p)
				setattr(module, name_p, fwAD.make_dual(p, g[start:start+p.numel()].view_as(p)))
				start += p.numel()
			# with torch.cuda.amp.autocast():
			output, Jv = fwAD.unpack_dual(model(x_batch))
			Jvs.append(Jv)
	Jvs = torch.stack(Jvs, 1)
	if y_batch is None:
		y_batch = output.argmax(-1)
	r = output.softmax(-1) - F.one_hot(y_batch, args.num_classes)
	K = (Jvs @ r.unsqueeze(-1)).squeeze()

	if not return_more:
		return K

	a = r.unsqueeze(-1) @ r.unsqueeze(1)
	a = torch.linalg.pinv(a) @ r.unsqueeze(-1)
	return K, output, a.squeeze()

@torch.enable_grad()
def batch_grad(model, x_batch, y_batch, args):
	g_batch = []
	for i, (x, y) in enumerate(zip(x_batch, y_batch)):
		o = model(x.unsqueeze(0))
		model.zero_grad()
		F.cross_entropy(o, y[None]).backward()
		g = torch.cat([p.grad.flatten() for p in model.parameters()])#.cpu()
		g_batch.append(g)
	return torch.stack(g_batch)

def lla_test(test_loader, model, device, args, Psi, cov_inv, verbose=True):
	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x, y in tqdm(test_loader, desc='Testing', total=len(test_loader)):
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			Psi_x, y_pred = Psi(x, return_logits=True)
			F_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)
			F_samples = (psd_safe_cholesky(F_var) @ torch.randn(F_var.shape[0], F_var.shape[1], args.num_samples_eval, device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + y_pred
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
		print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))
	return loss, acc, ece

def test(test_loader, model, device, args):
	model.eval()

	targets, confidences, predictions = [], [], []
	loss, acc, num_data = 0, 0, 0
	with torch.no_grad():
		for x_batch, y_batch in tqdm(test_loader, desc='Testing', total=len(test_loader)):
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

	print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))


if __name__ == '__main__':
	main()
