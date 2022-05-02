import warnings
import numpy as np
import os
import time
from tqdm import tqdm
import copy
import math
import gc

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

from functools import partial
import threading

from data import subsample

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class _ECELoss(torch.nn.Module):
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

		bin_boundaries_plot = torch.linspace(0, 1, 11)
		self.bin_lowers_plot = bin_boundaries_plot[:-1]
		self.bin_uppers_plot = bin_boundaries_plot[1:]

	def forward(self, confidences, predictions, labels, title=None):
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=confidences.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		accuracy_in_bin_list = []
		for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			accuracy_in_bin = 0
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean().item()
			accuracy_in_bin_list.append(accuracy_in_bin)

		if title:
			fig = plt.figure(figsize=(8,6))
			p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
			p2 = plt.plot([0,1], [0,1], '--', color='gray')

			plt.ylabel('Accuracy', fontsize=18)
			plt.xlabel('Confidence', fontsize=18)
			#plt.title(title)
			plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.xlim(left=0,right=1)
			plt.ylim(bottom=0,top=1)
			plt.grid(True)
			#plt.legend((p1[0], p2[0]), ('Men', 'Women'))
			plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=18)
			fig.tight_layout()
			plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')

		return ece

def psd_safe_eigen(K):
	Kprime = K.clone()
	jitter = 0
	jitter_new = None
	while True:
		p, q = torch.linalg.eigh(Kprime)
		if (p > 0).all():
			if jitter_new is not None:
				warnings.warn(
					f"K not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
			return p, q
		else:
			if jitter == 0:
				jitter_new = 1e-10
			else:
				jitter_new = jitter * 10
		Kprime.diagonal().add_(jitter_new - jitter)
		jitter = jitter_new


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.linalg.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise NanError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(6):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		raise e

@torch.enable_grad()
def jac(model, xs, ys, num_classes, full=False, dtype=torch.float32):
	I = torch.eye(num_classes).to(xs.device)
	Js = torch.zeros(xs.shape[0] * (num_classes if full else 1),
					 count_parameters(model), dtype=dtype, device=xs.device)
	for i, (x, y) in enumerate(zip(xs, ys)):
		o = model(x.unsqueeze(0))
		model.zero_grad()
		if full:
			for j in range(num_classes):
				o.backward(I[j].view(1, -1), retain_graph = False
					if j == num_classes - 1 else True)
				g = torch.cat([p.grad.flatten() for p in model.parameters()])
				Js[i * num_classes + j] = g
				model.zero_grad()
		else:
			grad_in = I[y.item()].view(1, -1)
			o.backward(grad_in)
			g = torch.cat([p.grad.flatten() for p in model.parameters()])
			Js[i] = g
	return Js

def build_dual_params_list(model, params, x_subsample, y_subsample, args, num_batches=1, verbose=True):
	indices = torch.rand(x_subsample.shape[0], args.num_classes).argmax(1).long() if args.random else y_subsample
	if num_batches == 1:
		Js = jac(model, x_subsample, indices, args.num_classes)
		mat = Js @ Js.T
	else:
		mat = torch.empty(x_subsample.shape[0], x_subsample.shape[0], device=x_subsample.device)
		s = x_subsample.shape[0] // num_batches
		for i in tqdm(range(0, x_subsample.shape[0], s), desc='Doing matmul', total=num_batches):
			ii = min(i + s, x_subsample.shape[0])
			Js_b = jac(model, x_subsample[i:ii], indices[i:ii], args.num_classes)

			for j in range(0, x_subsample.shape[0], s):
				jj = min(j + s, x_subsample.shape[0])
				Js_b2 = jac(model, x_subsample[j:jj], indices[j:jj], args.num_classes)
				mat[i:ii, j:jj] = Js_b @ Js_b2.T
				# print(i, j, torch.dist(Js_b, Js1[i:ii]), torch.dist(Js_b2, Js1[j:jj]), torch.dist(mat[i:ii, j:jj], mat1[i:ii, j:jj]))
				# print(mat[i:ii, j:jj], mat1[i:ii, j:jj])
		del Js_b2

	p, q = psd_safe_eigen(mat)
	p = p[range(-1, -(args.K+1), -1)]
	q = q[:, range(-1, -(args.K+1), -1)]
	tmp = q.div(p.sqrt())
	p = (p / x_subsample.shape[0])

	if num_batches == 1:
		V = Js.T @ tmp
	else:
		V = torch.zeros(count_parameters(model), args.K, device=tmp.device)
		s = x_subsample.shape[0] // num_batches
		for i in tqdm(range(0, x_subsample.shape[0], s), desc='Doing matmul', total=num_batches):
			ii = min(i + s, x_subsample.shape[0])
			Js_b = jac(model, x_subsample[i:ii], indices[i:ii], args.num_classes)
			V += Js_b.T @ tmp[i:ii]

	if verbose:
	   print('eigenvalues: ', p)
	   # print(V.norm(dim=1, p=2))

	dual_params_list = []
	for item in V.T:
		dual_params = {}
		start = 0
		for name, param in params.items():
			dual_params[name] = item[start:start+param.numel()].view_as(param) #.to(param.device)
			start += param.numel()
		dual_params_list.append(dual_params)
	return dual_params_list


def find_module_by_name(model, name):
	names = name.split(".")
	module = model
	for n in names[:-1]:
		module = getattr(module, n)
	return module, names[-1]

@torch.no_grad()
def Psi_raw(model, params, dual_params_list, x_batch, return_output=False):
	with fwAD.dual_level():
		Jvs = []
		for dual_params in dual_params_list:
			for name, param in params.items():
				module, name_p = find_module_by_name(model, name)
				delattr(module, name_p)
				setattr(module, name_p, fwAD.make_dual(param, dual_params[name]))
			# with torch.cuda.amp.autocast(): # not supported yet
			output, Jv = fwAD.unpack_dual(model(x_batch))
			Jvs.append(Jv)
	Jvs = torch.stack(Jvs, -1)
	if return_output:
		return Jvs, output
	else:
		return Jvs


class ConvNet(nn.Module):
	def __init__(self, num_classes=10):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Linear(7 * 7 * 32, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out

def check_approx_error(args, model, params, train_loader_noaug, val_loader, device, total=2000):
	if os.path.isfile(args.save_dir + '/approx_errors.npy'):
		results = np.load(args.save_dir + '/approx_errors.npy')
		draw_matrix(args.save_dir, results[:10], 'approx_error_ntk.pdf')
		draw_matrix(args.save_dir, results[10:], 'approx_error_ella_cov.pdf')
		return

	model2 = copy.deepcopy(model)
	x_full, y_full = subsample(train_loader_noaug, args.num_classes,
							   total+300, args.balanced, device)
	x_test, y_test = x_full[total:total+256], y_full[total:total+256]
	x_full, y_full = x_full[:total], y_full[:total]

	jac_full = jac(model, x_full, y_full, args.num_classes, full=True)
	jac_test = jac(model, x_test, y_test, args.num_classes, full=True).view(
		x_test.shape[0], args.num_classes, -1)

	K1 = jac_full @ jac_full.T
	K1_norm = torch.linalg.norm(K1, ord=2)

	with torch.no_grad():
		prob_full = model(x_full).softmax(-1).cpu()
		Lambda_full = prob_full.diag_embed() - prob_full[:, :, None] * prob_full[:, None, :]

	JLJ = torch.einsum('NCP,NCD,NDQ->PQ', jac_full.view(total, args.num_classes, -1),
		Lambda_full, jac_full.view(total, args.num_classes, -1))
	Sigma1 = JLJ + torch.eye(JLJ.shape[0]) / args.sigma2
	Sigma1 = Sigma1.inverse()
	# Sigma1_norm = torch.linalg.norm(Sigma1, ord=2)

	# eq 5
	pred1 = jac_test @ Sigma1 @ jac_test.permute(0, 2, 1)

	results = np.zeros((20, 10))
	for i, M in enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000]):
		J_x_subsample = jac(model, x_full[:M], y_full[:M],
							args.num_classes, random=args.random)
		for j, K in enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000]):
			if K > M:
				continue
			dual_params_list = build_dual_params_list(J_x_subsample, K, params)
			Psi = partial(Psi_raw, model2, params, dual_params_list)

			psi_full = Psi(x_full).cpu()
			psi_test = Psi(x_test).cpu()

			# nystrom approximation of K1
			K2 = psi_full.flatten(0, 1) @ psi_full.flatten(0, 1).T

			# normalized E'
			results[i, j] = torch.linalg.norm(K1 - K2, ord=2) / K1_norm

			# Eq 8
			G = torch.einsum('NCP,NCD,NDQ->PQ', psi_full, Lambda_full, psi_full) + torch.eye(psi_full.shape[-1]) / args.sigma2
			pred2 = psi_test @ G.inverse() @ psi_test.permute(0, 2, 1)
			results[i+10, j] = (torch.linalg.norm(pred1 - pred2, ord=2, dim=(-2, -1)) / torch.linalg.norm(pred1, ord=2, dim=(-2, -1))).mean()

			print(i, j, results[i, j], results[i+10, j])

			if 0 and K == M:
				# the explicit nystrom approximation
				K3 = jac_full @ J_x_subsample.T @ (J_x_subsample @ J_x_subsample.T).pinverse() @ (J_x_subsample @ jac_full.T)
				print('check1', torch.dist(K2, K3), K2[:5,:5], K3[:5,:5])

				# Sigma' in eq 21
				Sigma2 = (J_x_subsample @ JLJ @ J_x_subsample.T + (J_x_subsample @ J_x_subsample.T) / args.sigma2).inverse()
				Sigma2 = J_x_subsample.T @ Sigma2 @ J_x_subsample

				# another form of Sigma' in Appendix A.4
				P = J_x_subsample.T @ (J_x_subsample @ J_x_subsample.T).inverse() @ J_x_subsample
				Sigma3 = (P @ JLJ @ P + torch.eye(JLJ.shape[0]) / args.sigma2).inverse() + args.sigma2 * (P - torch.eye(JLJ.shape[0]))
				print('check2', torch.dist(Sigma2, Sigma3), Sigma2[:5, :5], Sigma3[:5, :5])

				# eq 21 equals to eq 8
				pred3 = jac_test @ Sigma2 @ jac_test.permute(0, 2, 1)
				print('check3', torch.dist(pred2, pred3), pred2[5], pred3[5])

				# E
				E = torch.linalg.norm(Sigma1 - Sigma2, ord=2)
				print('E', torch.dist(Sigma1, Sigma2), E, Sigma1_norm, E / Sigma1_norm)
				print(Sigma1[:5, :5], Sigma2[:5, :5])

	np.save(args.save_dir + '/approx_errors', results)
	draw_matrix(args.save_dir, results[:10], 'approx_error_ntk.pdf')
	draw_matrix(args.save_dir, results[10:], 'approx_error_ella_cov.pdf')

def draw_matrix(save_dir, matrix, title):
	ma = matrix.max()
	mi = 0

	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111)
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	im = ax.imshow(matrix, cmap='YlGn', vmin=mi, vmax=ma)
	ax.set_xlabel('$K$')
	ax.set_ylabel('$M$')

	ax.set_xticks(range(10))
	ax.set_xticklabels(map(str, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000]))
	ax.set_yticks(range(10))
	ax.set_yticklabels(map(str, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000]))

	ax.spines['bottom'].set_color('gray')
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.spines['left'].set_color('gray')
	ax.set_axisbelow(True)

	fig.subplots_adjust(right=0.95)
	cbar_ax = fig.add_axes([0.98, 0.15, 0.01, 0.8])
	fig.colorbar(im, cax=cbar_ax)

	fig.tight_layout()
	fig.savefig(os.path.join(save_dir, title), format='pdf', dpi=1000, bbox_inches='tight')

def draw_vector(save_dir, y, title):
	fig = plt.figure(figsize=(7, 5))
	ax = fig.add_subplot(111)
	ax.tick_params(axis='y', which='major', labelsize=12)
	ax.tick_params(axis='y', which='minor', labelsize=12)
	ax.tick_params(axis='x', which='major', labelsize=12)
	ax.tick_params(axis='x', which='minor', labelsize=12)

	x = np.arange(len(y))
	b = ax.bar(x, y)

	ax.set_xticks(x)
	ax.set_xticklabels(map(str, [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000]))
	# ax.set_yscale('log')
	ax.set_xlabel('$M(=K)$')
	ax.set_ylabel('$\mathcal{E}/\Vert\mathcal{\Sigma}\Vert$')

	ax.spines['bottom'].set_color('gray')
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.spines['left'].set_color('gray')

	fig.tight_layout()
	fig.savefig(os.path.join(save_dir, title), format='pdf', dpi=1000, bbox_inches='tight')

def measure_speed(model, params, dual_params_list, model_bk, x, y, runs=500):
	torch.cuda.current_stream().synchronize()
	t2 = time.time()
	with torch.no_grad():
		with fwAD.dual_level():
			dual_params = dual_params_list[0]
			for name, param in params.items():
				module, name_p = find_module_by_name(model, name)
				delattr(module, name_p)
				setattr(module, name_p, fwAD.make_dual(param, dual_params[name]))
			for _ in range(500):
				model(x)
	torch.cuda.current_stream().synchronize()
	t3 = time.time()

	torch.cuda.current_stream().synchronize()
	t0 = time.time()
	with torch.no_grad():
		for _ in range(500):
			model_bk(x)
	torch.cuda.current_stream().synchronize()
	t1 = time.time()

	print("Time cost comparison {:.4f} vs. {:.4f} ({:.4f}K times)".format(
		(t3-t2)/500, (t1-t0)/500, (t3-t2)/(t1-t0)))

def do_dot(a, b, out):
	out[:] = torch.matmul(a, b)

# def parallel_mm(a, b, nblocks, mblocks=1, use_gpu=False):
# 	"""
# 	Return the matrix product a @ b.
# 	"""
# 	bT = b.T
# 	# assert a.shape[0] % nblocks == 0 and bT.shape[0] % mblocks == 0
# 	s = a.shape[0]//nblocks
# 	t = bT.shape[0]//mblocks
#
# 	#a_blocks = a.view(nblocks, s, a.shape[1])
# 	#bT_blocks = bT.view(mblocks, t, bT.shape[1])
# 	out = torch.empty((a.shape[0], bT.shape[0]))
# 	if use_gpu:
# 		for i in tqdm(range(0, a.shape[0], s), desc='Doing matmul', total=nblocks):
# 			m1 = a[i:min(i + s, a.shape[0])].cuda(non_blocking=True).float()
# 			for j in range(0, bT.shape[0], t):
# 				with torch.no_grad():
# 					m2 = bT[j:min(j + t, bT.shape[0])].cuda(non_blocking=True).float()
# 				out[i:min(i + s, a.shape[0]), j:min(j + t, bT.shape[0])] = (m1 @ m2.T).cpu()
# 		del m1, m2
# 	else:
# 		threads = []
# 		for i in tqdm(range(0, a.shape[0], s), desc='Doing matmul', total=nblocks):
# 			for j in range(0, bT.shape[0], t):
# 				th = threading.Thread(target=do_dot,
# 									  args=(a[i:min(i + s, a.shape[0])].float(),
# 											bT[j:min(j + t, bT.shape[0])].float().T,
# 											out[i:min(i + s, a.shape[0]), j:min(j + t, bT.shape[0])]))
# 				th.start()
# 				threads.append(th)
#
# 		for th in threads:
# 			th.join()
#
# 	return out

if __name__ == '__main__':
	a = torch.randn(3333, 1111)

	b = torch.randn(1111, 222)

	start = time.time()
	r1 = parallel_mm(a, b, 13, 13, True)
	time_par = time.time() - start
	print('parallel_mm: {:.2f} seconds taken'.format(time_par))

	start = time.time()
	r2 = torch.matmul(a, b)
	time_dot = time.time() - start
	print('torch.matmul: {:.2f} seconds taken'.format(time_dot))

	# print(r1[:10, :10])
	# print(r2[:10, :10])
	# print(r1[-10:, -10:])
	# print(r2[-10:, -10:])

	assert torch.allclose(r1, r2, atol=1e-4), 'dist is {}/{}'.format(torch.dist(r1, r2).item(), torch.dist(r1, torch.zeros_like(r1)).item())
