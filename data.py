import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def subsample(loader, num_classes, subsample_number, balanced, device, verbose=False):
	xs, ys = [], []
	cnt = np.zeros(num_classes)
	for x_batch, y_batch in loader:
		for x, y in zip(x_batch, y_batch):
			if np.all(balanced and cnt >= subsample_number//num_classes) or \
				((not balanced) and cnt.sum() >= subsample_number):
				xs = torch.stack(xs)
				ys = torch.stack(ys)
				if verbose:
					print("The frequency of the sampled labels")
					print(np.unique(ys.numpy(), return_counts=True))
				return xs.to(device), ys.to(device)
			if balanced and cnt[y.item()] >= subsample_number//num_classes:
				continue
			xs.append(x); ys.append(y)
			cnt[y.item()] += 1

def data_loaders(args, valid_size=None, noaug=None):
	if 'cifar' in args.dataset:
		return cifar_loaders(args, valid_size, noaug)
	elif args.dataset == 'mnist':
		return mnist_loaders(args, valid_size, noaug)
	else:
		return imagenet_loaders(args, valid_size, noaug)

def mnist_loaders(args, valid_size=None, noaug=None):
	dset = datasets.MNIST
	T = transforms.Compose([
		transforms.ToTensor()])

	test_loader = torch.utils.data.DataLoader(
		dset(root=args.data_root, train=False, transform=T, download=True),
		batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	train_dataset = dset(root=args.data_root, train=True, transform=T, download=True)
	if valid_size is not None:
		valid_dataset = dset(root=args.data_root, train=True, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(28, 4),
			transforms.ToTensor(),
			Cutout(16),
		]), download=True)
		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))
		np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
		valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, sampler=train_sampler,
			num_workers=args.workers, pin_memory=True)
		val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
			num_workers=args.workers, pin_memory=True)
		return train_loader, val_loader, test_loader
	else:
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)
		return train_loader, test_loader

def cifar_loaders(args, valid_size=None, noaug=None):
	if args.dataset == 'cifar10':
		normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
										 std=[0.2023, 0.1994, 0.201])
		dset = datasets.CIFAR10
	elif args.dataset == 'cifar100':
		normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
										 std=[0.2673, 0.2564, 0.2761])
		dset = datasets.CIFAR100
	else:
		raise NotImplementedError

	if noaug:
		T = transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])
	else:
		T = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		])

	T_val = transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])

	test_loader = torch.utils.data.DataLoader(
		dset(root=args.data_root, train=False, transform=T_val, download=True),
		batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	train_dataset = dset(root=args.data_root, train=True, transform=T, download=True)
	if valid_size is not None:
		valid_dataset = dset(root=args.data_root, train=True, transform=transforms.Compose([
			# transforms.RandomHorizontalFlip(),
			# transforms.RandomCrop(32, 4),
			# transforms.ToTensor(),
			# normalize,
			# Cutout(16),
			transforms.RandomResizedCrop(size=32, scale=(0.6 if args.arch == 'cifar10_resnet44' else 0.5, 1.)),
			# transforms.RandomGrayscale(p=0.2),
			# transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			# Cutout(2),
		]), download=True)
		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))
		np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
		valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, sampler=train_sampler,
			num_workers=args.workers, pin_memory=False)
		val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
			num_workers=args.workers, pin_memory=False)

		# remove the randomness
		xs, ys = [], []
		for _ in range(1):
			for x, y in val_loader:
				xs.append(x); ys.append(y)
		xs = torch.cat(xs); ys = torch.cat(ys)
		valid_dataset = torch.utils.data.TensorDataset(xs, ys)
		val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)

		return train_loader, val_loader, test_loader
	else:
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=False)
		return train_loader, test_loader

class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


def imagenet_loaders(args, valid_size=None, noaug=None):

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	if noaug:
		T = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])
	else:
		T = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])

	T_val = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])

	test_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(os.path.join(args.data_root, 'val'), transform=T_val),
		batch_size=args.test_batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	train_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=T)
	if valid_size is not None:
		valid_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'),
			transform=create_transform(
				input_size=224,
				scale=(0.08, 0.1),
				is_training=True,
				color_jitter=0.4,
				auto_augment=None, #'original', #'v0' #'rand-m9-mstd0.5-inc1', #'v0', 'original'
				interpolation='bicubic',
				re_prob=0.25, #0.25,
	            re_mode='pixel',
	            re_count=1,
				mean=IMAGENET_DEFAULT_MEAN,
				std=IMAGENET_DEFAULT_STD,
			)
		)
		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))
		np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
		valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, sampler=train_sampler,
			num_workers=args.workers, pin_memory=False)
		val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
			num_workers=args.workers, pin_memory=False)

		xs, ys = [], []
		for _ in range(1):
			for x, y in val_loader:
				xs.append(x); ys.append(y)
		xs = torch.cat(xs); ys = torch.cat(ys)
		valid_dataset = torch.utils.data.TensorDataset(xs, ys)
		val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)

		return train_loader, val_loader, test_loader
	else:
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=False)
		return train_loader, test_loader
