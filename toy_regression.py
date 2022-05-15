import math
from tqdm import tqdm
import scipy as sp
import random
import copy
import time
from functools import partial

# for NN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import pandas as pd
import numpy.random as rnd
import seaborn as sns

from laplace import Laplace
from laplace.curvature import BackPackGGN

from utils import build_dual_params_list, Psi_raw, psd_safe_cholesky

class Erf(torch.nn.Module):
    def __init__(self):
        super(Erf, self).__init__()

    def forward(self, x):
        return x.erf()

class Idt(torch.nn.Module):
    def __init__(self):
        super(Idt, self).__init__()

    def forward(self, x):
        return x

def fn_make_NN(activation_fn, n_hidden=[100], dropout=0.):
    D_in, D_out = 1, 1 # input and output dimension

    if activation_fn == 'relu':
        mid_act = torch.nn.ReLU
    elif activation_fn == 'erf':
        mid_act = Erf
    elif activation_fn == 'idt':
        mid_act = Idt
    elif activation_fn == 'sig':
        mid_act = torch.nn.Sigmoid
    elif activation_fn == 'tanh':
        mid_act = torch.nn.Tanh

    layers = [torch.nn.Linear(D_in, n_hidden[0]),
        mid_act(),
        torch.nn.Linear(n_hidden[-1], D_out, bias=False)]
    for n_hidden1, n_hidden2 in zip(n_hidden[:-1], n_hidden[1:]):
        layers.insert(-1, torch.nn.Linear(n_hidden1, n_hidden2))
        layers.insert(-1, mid_act())
    model = torch.nn.Sequential(*layers)
    return model

random.seed(1000)
torch.manual_seed(1000)
np.random.seed(1000)
device = torch.device('cpu')

x_train = np.atleast_2d(np.concatenate([np.linspace(-1.5,-0.6,8), np.linspace(0.6,1.5,8)])).T
y_train =  np.sin(x_train*2) + np.random.normal(loc=0.,scale=0.2, size=x_train.shape)
# y_train[-1] = y_train[-1]-0.6
x_test_np = np.atleast_2d(np.linspace(-3, 3, 100)).T
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test_np).float().to(device)
xlim = [-3, 3]
ylim = [-3, 3]
dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=x_train.shape[0], pin_memory=False)

n_hidden = [2, 32, 512] 	# no. hidden units in NN
data_noise = 0.01 # estimated noise variance
activation_fn = 'tanh'

sigma2 = 1
epochs = 1000
l_rate = 1e-5
model = fn_make_NN(activation_fn, n_hidden=n_hidden).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=l_rate, weight_decay=0, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

print(model)
for _ in tqdm(range(epochs)):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = (model(x) - y).norm() ** 2 / 2. / data_noise
        mean = parameters_to_vector(model.parameters())
        reg = mean.norm() ** 2 / 2. / sigma2
        optimizer.zero_grad()
        (loss+reg).backward()
        optimizer.step()
    scheduler.step()

model.eval()
with torch.no_grad():
    y_pred_map = model(x_test).cpu().numpy()

fig = plt.figure(figsize=(25, 4))

# ax = fig.add_subplot(141)
# ax.plot(x_test_np, y_pred_map, 'b-', linewidth=2.,label=u'Prediction')
# ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
# ax.set_ylim(*ylim)
# ax.set_xlim(*xlim)
# plt.xticks([])
# plt.yticks([])

model_bk = copy.deepcopy(model)
params = {name: p for name, p in model.named_parameters()}
dual_params_list = build_dual_params_list(model, params,
                                          x_train, y_train,
                                          num_classes=1, random=True,
                                          K=5,
                                          verbose=False)
Psi = partial(Psi_raw, model, params, dual_params_list)
with torch.no_grad():
    Psi_x = Psi(x_train)
    cov = torch.einsum('bok,bol->kl', Psi_x, Psi_x) / data_noise
    cov.diagonal().add_(1 / sigma2)
    cov_inv = cov.inverse()

    Psi_x, y_pred_mu = Psi(x_test, return_output=True)
    y_pred_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)

    y_pred_mu, y_pred_std = y_pred_mu.squeeze().cpu().numpy(), y_pred_var.sqrt().clamp(min=1e-4).squeeze().cpu().numpy()
ax = fig.add_subplot(151)
ax.plot(x_test_np, y_pred_mu, 'b-', linewidth=2.,label=u'Prediction')
ax.plot(x_test_np, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
ax.fill_between(x_test_np[:,0], y_pred_mu + 2 * y_pred_std, y2=y_pred_mu - 2 * y_pred_std, alpha=1, fc='lightskyblue', ec='None')
ax.fill_between(x_test_np[:,0], y_pred_mu + 1 * y_pred_std, y2=y_pred_mu - 1 * y_pred_std, alpha=1, fc='deepskyblue', ec='None')
ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
plt.xticks([])
plt.yticks([])
ax.set_title('ELLA',y=-0.02,pad=-14)

model = model_bk

la = Laplace(copy.deepcopy(model), 'regression', sigma_noise=data_noise,
             prior_precision=1/sigma2,
             subset_of_weights='all',
             hessian_structure='full')
la.fit(train_loader)
with torch.no_grad():
    y_pred_mu, y_pred_var = la(x_test)
    y_pred_mu, y_pred_std = y_pred_mu.squeeze().cpu().numpy(), y_pred_var.sqrt().clamp(min=1e-4).squeeze().cpu().numpy()
ax = fig.add_subplot(152)
ax.plot(x_test_np, y_pred_mu, 'b-', linewidth=2.,label=u'Prediction')
ax.plot(x_test_np, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
ax.fill_between(x_test_np[:,0], y_pred_mu + 2 * y_pred_std, y2=y_pred_mu - 2 * y_pred_std, alpha=1, fc='lightskyblue', ec='None')
ax.fill_between(x_test_np[:,0], y_pred_mu + 1 * y_pred_std, y2=y_pred_mu - 1 * y_pred_std, alpha=1, fc='deepskyblue', ec='None')
ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
plt.xticks([])
plt.yticks([])
ax.set_title('LLA',y=-0.02,pad=-14)

la = Laplace(copy.deepcopy(model), 'regression', sigma_noise=data_noise,
             prior_precision=1/sigma2,
             subset_of_weights='all',
             hessian_structure='kron')
la.fit(train_loader)
with torch.no_grad():
    y_pred_mu, y_pred_var = la(x_test)
    y_pred_mu, y_pred_std = y_pred_mu.squeeze().cpu().numpy(), y_pred_var.sqrt().clamp(min=1e-4).squeeze().cpu().numpy()
ax = fig.add_subplot(153)
ax.plot(x_test_np, y_pred_mu, 'b-', linewidth=2.,label=u'Prediction')
ax.plot(x_test_np, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
ax.fill_between(x_test_np[:,0], y_pred_mu + 2 * y_pred_std, y2=y_pred_mu - 2 * y_pred_std, alpha=1, fc='lightskyblue', ec='None')
ax.fill_between(x_test_np[:,0], y_pred_mu + 1 * y_pred_std, y2=y_pred_mu - 1 * y_pred_std, alpha=1, fc='deepskyblue', ec='None')
ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
plt.xticks([])
plt.yticks([])
ax.set_title('LLA-KFAC',y=-0.02,pad=-14)

la = Laplace(copy.deepcopy(model), 'regression', sigma_noise=data_noise,
             prior_precision=1/sigma2,
             subset_of_weights='all',
             hessian_structure='diag')
la.fit(train_loader)
with torch.no_grad():
    y_pred_mu, y_pred_var = la(x_test)
    y_pred_mu, y_pred_std = y_pred_mu.squeeze().cpu().numpy(), y_pred_var.sqrt().clamp(min=1e-4).squeeze().cpu().numpy()
ax = fig.add_subplot(154)
ax.plot(x_test_np, y_pred_mu, 'b-', linewidth=2.,label=u'Prediction')
ax.plot(x_test_np, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
ax.fill_between(x_test_np[:,0], y_pred_mu + 2 * y_pred_std, y2=y_pred_mu - 2 * y_pred_std, alpha=1, fc='lightskyblue', ec='None')
ax.fill_between(x_test_np[:,0], y_pred_mu + 1 * y_pred_std, y2=y_pred_mu - 1 * y_pred_std, alpha=1, fc='deepskyblue', ec='None')
ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
plt.xticks([])
plt.yticks([])
ax.set_title('LLA-Diag',y=-0.02,pad=-14)

la = Laplace(copy.deepcopy(model), 'regression', sigma_noise=data_noise,
             prior_precision=1/sigma2,
             subset_of_weights='last_layer',
             hessian_structure='full')
la.fit(train_loader)
with torch.no_grad():
    y_pred_mu, y_pred_var = la(x_test)
    y_pred_mu, y_pred_std = y_pred_mu.squeeze().cpu().numpy(), y_pred_var.sqrt().clamp(min=1e-4).squeeze().cpu().numpy()
ax = fig.add_subplot(155)
ax.plot(x_test_np, y_pred_mu, 'b-', linewidth=2.,label=u'Prediction')
ax.plot(x_test_np, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
ax.plot(x_test_np, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
ax.fill_between(x_test_np[:,0], y_pred_mu + 2 * y_pred_std, y2=y_pred_mu - 2 * y_pred_std, alpha=1, fc='lightskyblue', ec='None')
ax.fill_between(x_test_np[:,0], y_pred_mu + 1 * y_pred_std, y2=y_pred_mu - 1 * y_pred_std, alpha=1, fc='deepskyblue', ec='None')
ax.plot(x_train[:,0], y_train, 'r.', markersize=12, markeredgecolor='k', markeredgewidth=0.5)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')
plt.xticks([])
plt.yticks([])
ax.set_title('LLA$^*$',y=-0.02,pad=-14)

fig.savefig('toy_regression.pdf', format='pdf', dpi=1000, bbox_inches='tight')
