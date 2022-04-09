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

from data import data_loaders, subsample
from utils import count_parameters, _ECELoss, psd_safe_eigen, psd_safe_cholesky, find_module_by_name

from MDEQVision.lib.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='LLA for DEQ')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--not-use-gpu', action='store_true', default=False)

    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--test-batch-size', default=128, type=int)

    parser.add_argument('--arch', type=str, default='mdeq')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data-root', type=str, default='./data/cifar10')
    parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                        help='path to pretrained MAP checkpoint (default: none)')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='./logs/', type=str)
    parser.add_argument('--job-id', default='default', type=str)


    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--subsample-number', default=100, type=int,
                        metavar='N', help='subsample data')
    parser.add_argument('--balanced', action='store_true', default=False)
    parser.add_argument('--only-target-jacobian', action='store_true', default=False)

    parser.add_argument('--early-stop', default=None, type=int)
    parser.add_argument('--search-freq', default=None, type=int)
    parser.add_argument('--sigma2', default=1e8, type=float)

    parser.add_argument('--num-samples-eval', default=256, type=int,
                        metavar='N')
    parser.add_argument('--ntk-std-scale', default=1, type=float)

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def wn(model):
    # This is the multiscale-deq version of weight norm
    # I moved it here so that it would not have to be inside forward
    def _norm(p, dim):
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    def compute_weight(module=nn.Conv2d, name='weight'):
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')
        return v * (g / _norm(v, 0))
    for k in model.state_dict().keys():
        if "weight_g" in k:
            k = k.replace(".0", "[0]").replace(".1", "[1]").replace(".2", "[2]").replace(".3", "[3]").replace(".weight_g", "")
            con = "model." + k
            setattr(eval(con), 'weight', compute_weight(eval(con), 'weight'))

def main():
    args = parse_args()
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
    elif args.arch == "mdeq":
        from MDEQVision.lib.models.mdeq import get_cls_net
        model = get_cls_net(config).float().to(device)
    else:
        model = nn.Sequential(nn.Flatten(1), nn.Linear(3072, args.num_classes))

    params = {name: p for name, p in model.named_parameters()}
    print("Number of parameters", count_parameters(model))
    if args.pretrained is not None:
        print("Load MAP model from", args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))
    if args.arch == "mdeq":
        wn(model)

    model.eval()

    print("---------MAP model ---------")
    test(test_loader, model, device, args)

    ###### lla ######
    ## collect a set of Jacobians
    x_subsample, y_subsample = subsample(train_loader_noaug, args)
    # print(np.unique(y_subsample.numpy(), return_counts=True))
    x_subsample, y_subsample = x_subsample.to(device), y_subsample.to(device)
    g_subsample = batch_grad(model, x_subsample, y_subsample, args)
    K = torch.einsum('np,mp->nm', g_subsample, g_subsample)

    p, q = psd_safe_eigen(K)
    p = p[range(-1, -(args.k+1), -1)]
    q = q[:, range(-1, -(args.k+1), -1)]
    q = (g_subsample.T @ q / p.sqrt()).T
    p = (p / g_subsample.shape[0])
    print(p)
    print(q.norm(dim=1, p=2))
    del g_subsample

    ## convert q into fwAD model's parameters
    dual_params_list = []
    for item in q:
        dual_params = {}
        start = 0
        for name, param in params.items():
            dual_params[name] = item[start:start+param.numel()].view_as(param).to(param.device)
            start += param.numel()
        dual_params_list.append(dual_params)

    Psi = partial(Psi_raw, model, params, dual_params_list)

    ## check if the approximation is correct
    # print((Psi(x_subsample).permute(0, 2, 1).flatten(0, 1) @ Psi(x_subsample).permute(0, 2, 1).flatten(0, 1).T)[:10, :10])
    # print(K[:10, :10])
    # print(torch.dist(Psi(x_subsample).flatten(0, 1) @ Psi(x_subsample).flatten(0, 1).T, K.to(device)), torch.dist(K, torch.zeros_like(K)))

    ## pass the training set
    best_value = 1e8; best = None; best_test_results = None
    with torch.no_grad():
        cov = torch.zeros(args.k, args.k).cuda(non_blocking=True)
        for i, (x, y) in tqdm(enumerate(train_loader_noaug), desc='Passing the training set', total=len(train_loader_noaug)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            Psi_x, logits = Psi(x, return_output=True)
            prob = logits.softmax(-1)
            Delta_x = prob.diag_embed() - prob[:, :, None] * prob[:, None, :]
            cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x, Psi_x)

            if args.search_freq and (i + 1) * args.batch_size % args.search_freq == 0:
                cov_clone = cov.data.clone()
                cov_clone.diagonal().add_(1/args.sigma2)
                cov_inv = cov_clone.inverse()
                val_loss, _, _ = lla_test(test_loader, model, device, args, Psi, cov_inv, verbose=True)
                # if val_loss < best_value:
                # 	best_value = val_loss
                # 	best = (i + 1) * args.batch_size
                # 	test_loss, test_acc, test_ece = lla_test(test_loader, model, device, args, Psi, cov_inv, verbose=False)
                # 	best_test_results = "Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(test_loss, test_acc, test_ece)
                # print("Current subsample number: {}, loss: {:.4f}, best subsample number: {}, best loss: {:.4f}"
                # 	  "\n    {}".format((i + 1) * args.batch_size, val_loss, best, best_value, best_test_results))

            if args.early_stop and (i + 1) * args.batch_size >= args.early_stop:
                print("--------- LLA model ---------")
                cov.diagonal().add_(1/args.sigma2)
                cov_inv = cov.inverse()
                lla_test(test_loader, model, device, args, Psi, cov_inv)
                return

@torch.no_grad()
def Psi_raw(model, params, dual_params_list, x_batch, return_output=False):
    with fwAD.dual_level():
        Jvs = []
        for dual_params in dual_params_list:
            for name, param in params.items():
                module, name_p = find_module_by_name(model, name)
                delattr(module, name_p)
                setattr(module, name_p, fwAD.make_dual(param, dual_params[name]))
            # with torch.cuda.amp.autocast():
            output, Jv = fwAD.unpack_dual(model(x_batch))
            Jvs.append(Jv)
    Jvs = torch.stack(Jvs, -1)
    if return_output:
        return Jvs, output
    else:
        return Jvs

@torch.enable_grad()
def batch_grad(model, x_batch, y_batch, args):
    ones = torch.eye(args.num_classes).to(x_batch.device)
    g_batch = torch.zeros(x_batch.shape[0] * (1 if args.only_target_jacobian else args.num_classes), count_parameters(model))
    for i, (x, y) in tqdm(enumerate(zip(x_batch, y_batch)), desc='Estimating Jacobian', total=x_batch.shape[0]):
        o = model(x.unsqueeze(0))
        model.zero_grad()
        if args.only_target_jacobian:
            wn(model)
            o.backward(ones[y.item()].view(1, -1))
            g = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu()
            g_batch[i] = g
        else:
            for j in range(args.num_classes):
                o.backward(ones[j].view(1, -1), retain_graph = False if j == args.num_classes - 1 else True)
                g = torch.cat([p.grad.flatten() for p in model.parameters()]).cpu()
                g_batch[i * args.num_classes + j] = g
                model.zero_grad()
    return g_batch

def lla_test(test_loader, model, device, args, Psi, cov_inv, verbose=True):
    targets, confidences, predictions = [], [], []
    loss, acc, num_data = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Testing', total=len(test_loader)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            Psi_x, y_pred = Psi(x, return_output=True)
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
