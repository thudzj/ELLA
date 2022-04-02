from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import OrderedDict
import re
import math
import numpy as np
import argparse
import os
import pprint
import shutil
import sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import _init_paths
import models
from config import config
from config import update_config
from core.cls_function import train, validate

# from backpack import backpack, extend
# from backpack.extensions import BatchGrad

import torch.distributions as dists
from netcal.metrics import ECE
from laplace import Laplace
from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface, BackPackInterface, BackPackGGN, BackPackEF, AsdlInterface, AsdlGGN, AsdlEF, AsdlHessian

from torch.autograd.functional import jacobian
from torch.nn.utils import _stateless



def parse_args():
    parser = argparse.ArgumentParser(description='LLA for DEQ')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--not-use-gpu', action='store_true', default=False)

    parser.add_argument('-b', '--batch-size', default=200, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--test-batch-size', default=200, type=int)

    parser.add_argument('--dataset', type=str, default='test_cifar10')
    parser.add_argument('--data-root', type=str, default='./data/cifar10')
    parser.add_argument('--pretrained', default=None, type=str, metavar='PATH', #"./output/cifar10/cls_mdeq_LARGE_reg/model_best.pth.tar"
                        help='path to pretrained MAP checkpoint (default: none)')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='./logs/', type=str)
    parser.add_argument('--job-id', default='default', type=str)

    parser.add_argument('--nystrom-samples', default=100, type=int,
                        metavar='N', help='subsample size for nystrom')
    parser.add_argument('--sigma2', default=0.1, type=float)
    parser.add_argument('--num-samples-eval', default=256, type=int,
                        metavar='N', help='subsample size for nystrom')
    parser.add_argument('--ntk-std-scale', default=10, type=float)
    parser.add_argument('--use-normalizer', action='store_true', default=False)

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

def main():
    torch.manual_seed(42)
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda') if not args.not_use_gpu and torch.cuda.is_available() else torch.device('cpu')

    args.save_dir = os.path.join(args.save_dir, args.job_id)
    args.num_classes = 10 if (args.dataset == 'cifar10' or args.dataset == 'test_cifar10') else 100

    if os.path.isdir('/data/LargeData/cifar/'):
        args.data_root = '/data/LargeData/cifar/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



    from models.mdeq import get_cls_net
    model = get_cls_net(config).float().to(device)
    model.eval()
    model(torch.rand(1,3,32,32).to(device)).sum().backward()
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n)
    print("Number of parameters", count_parameters(model))
    if args.pretrained is not None:
        print("Load MAP model from", args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))


    train_loader, val_loader = cifar_loaders(args, batch_size=5)

    print("---------MAP model ---------")
    test(val_loader, model, device, args)

    start = time.time()
    targets = torch.cat([y for x, y in val_loader], dim=0).numpy()
    @torch.no_grad()
    def predict(dataloader, model, laplace=False):
        py = []

        for x, _ in dataloader:
            if laplace:
                py.append(model(x.cuda()))
            else:
                py.append(torch.softmax(model(x.cuda()), dim=-1))

        return torch.cat(py).cpu().numpy()

    probs_map = predict(val_loader, model, laplace=False)
    acc_map = (probs_map.argmax(-1) == targets).mean()
    ece_map = ECE(bins=15).measure(probs_map, targets)
    nll_map = 0.0 #-dists.Categorical(probs_map).log_prob(targets).mean()

    print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

    # Laplace
#     la = Laplace(model, 'classification',
#                  subset_of_weights='all',
#                  hessian_structure='diag',
#                  backend=AsdlGGN)
    la = Laplace(model, 'classification',
             subset_of_weights='all',
             hessian_structure='lowrank',
             )
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')

    probs_laplace = predict(val_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace, targets)
    nll_laplace = 0.0 #-dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
    print(time.time() - start)

    J_nystrom = torch.load("J.pt")

#     ## calculate eigenfunctions by the nystrom method
#     train_loader, val_loader = cifar_loaders(args, batch_size=args.nystrom_samples)
#     x_nystrom, _ = next(iter(train_loader))
#     x_nystrom = x_nystrom.to(device, non_blocking=True)
# #     J_nystrom = jacob(model, x_nystrom, args).cpu()
#     J_nystrom = torch.zeros(10, args.nystrom_samples, 11097438)
#     names = list(n for n, _ in model.named_parameters())

#     start = time.time()
#     for i in range(args.nystrom_samples // 5):

#         data = x_nystrom[5*i:5*(i+1)]
#         ja = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data), tuple(model.parameters()))
#         final = torch.cat([g.flatten(2) for g in ja], 2).detach().cpu().permute(1, 0, 2)
#         J_nystrom[:,5*i:5*(i+1),:] = final
#         del ja

# #     J_nystrom = jacob(bp_model, x_nystrom, args).permute(2, 0, 1)
# #     print(J_nystrom.shape)
# #     K = torch.einsum('onp,omp->onm', J_nystrom, J_nystrom)

# #     start = time.time()
# #     vs = []
# #     temp = torch.eye(10)
# #     for i in range(args.nystrom_samples):
# #         vs.append(temp)
# #     vs = torch.stack(vs, 0).to(device)
# #     with torch.enable_grad():
# #         JJ = torch.zeros(10, args.nystrom_samples, 11097438)
# #         for k in range(args.num_classes):
# #             J = torch.zeros(args.nystrom_samples, 11097438)
# #             for i in range(args.nystrom_samples):
# #                 output = model(torch.unsqueeze(x_nystrom[i], 0))
# #                 model.zero_grad()
# #                 if k != args.num_classes - 1:
# #                     result = torch.autograd.grad(output, model.parameters(), vs[i,k,:].view_as(output), retain_graph=True)
# #                 else:
# #                     result = torch.autograd.grad(output, model.parameters(), vs[i,k,:].view_as(output), retain_graph=False)
# #                 temp = torch.cat([g.flatten(0) for g in result], 0).detach()
# #                 J[i] = temp
# #             JJ[k] = J
# #     J_nystrom = JJ.cpu()


    K = torch.einsum('onp,omp->onm', J_nystrom, J_nystrom)

    if args.use_normalizer:
        normalizer = K.diagonal(dim1=1, dim2=2).mean() #-1
        K = K / normalizer#.view(-1, 1, 1)

    p, q = torch.linalg.eigh(K)
#     torch.save(p, "p.pt")
#     torch.save(q, "q.pt")
#     torch.save(J_nystrom, "J.pt")
    eigenvalues = p / args.nystrom_samples
    #
    if args.use_normalizer:
        Psi = lambda x: torch.einsum('bop,onp,onm->bom', jacob(model, x, args), J_nystrom, q) / p.add(1e-8).sqrt() / normalizer#.view(-1, 1)
        eigenfuncs = lambda x: torch.einsum('bop,onp,onm->bom', jacob(model, x, args), J_nystrom, q) / p.add(1e-8) * math.sqrt(args.nystrom_samples) / normalizer#.view(-1, 1)
    else:
        Psi = lambda x: torch.einsum('bop,onp,onm->bom', jacob(model, x, args), J_nystrom, q) / p.add(1e-8).sqrt()
        eigenfuncs = lambda x: torch.einsum('bop,onp,onm->bom', jacob(model, x, args), J_nystrom, q) / p.add(1e-8) * math.sqrt(args.nystrom_samples)

#     ## pass the training set
#     model.eval()
#     train_loader, val_loader = cifar_loaders(args, batch_size=5)
#     with torch.no_grad():
#         cov = torch.zeros(args.nystrom_samples, args.nystrom_samples)
#         for i, (x, _) in enumerate(train_loader):
#             x = x.to(device)
#             output = model(x)
#             prob = output.softmax(-1)
#             Delta_x = (prob.diag_embed() - prob[:, :, None] * prob[:, None, :])
#             Psi_x = Psi(x)
#             cov += torch.einsum('bok,boj,bjl->kl', Psi_x, Delta_x.detach().cpu(), Psi_x)
# #             del Delta_x
# #             del Psi_x
#             print("done with " + str(i) + " th batch")
#             print(time.time() - start)

# #             t = torch.cuda.get_device_properties(0).total_memory
# #             print(t)
# #             r = torch.cuda.memory_reserved(0)
# #             a = torch.cuda.memory_allocated(0)
# #             f = r-a  # free inside reserved
# #             print(f)
#         cov.diagonal().add_(1/args.sigma2)
#         cov_inv = cov.inverse()
#     print(cov_inv)
#     torch.save(cov, "cov.pt")
#     torch.save(cov_inv, "inv.pt")

    cov = torch.load("cov.pt")
    cov_inv = cov.inverse()
    print(cov_inv)

    ## test on validation data
    print("---------LLA model ---------")
    lla_test(val_loader, model, device, args, Psi, cov_inv)
#     print(time.time() - start)

def jacob(model, x, args):
    with torch.enable_grad():
        names = list(n for n, _ in model.named_parameters())
        ja = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x), tuple(model.parameters()))
        final = torch.cat([g.flatten(2) for g in ja], 2).detach().cpu()
        del ja
        return final
#         JJ = []
#         for k in range(args.num_classes):
#             J = []
#             output = bp_model(x)
#             bp_model.zero_grad()
#             if k != args.num_classes - 1:
#                 result = torch.autograd.grad(output, bp_model.parameters(), vs[k].view_as(output), retain_graph=True)
#             else:
#                 result = torch.autograd.grad(output, bp_model.parameters(), vs[k].view_as(output), retain_graph=False)
#             temp = torch.cat([g.flatten(0) for g in result], 0).detach().cpu()
#             J.append(temp)
#             temp = torch.stack(J, 0).detach().cpu()
#             JJ.append(temp)
#         return torch.stack(JJ, 0).permute(1, 0, 2).detach().cpu()

def lla_test(test_loader, model, device, args, Psi, cov_inv):
    model.eval()
    start = time.time()
    probs = []
    targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            Psi_x = Psi(x)
            F_var = Psi_x @ cov_inv.unsqueeze(0) @ Psi_x.permute(0, 2, 1)
            F_var = F_var.to(device)
            F_samples = (psd_safe_cholesky(F_var) @ torch.randn(F_var.shape[0], F_var.shape[1], args.num_samples_eval, device=F_var.device)).permute(2, 0, 1) * args.ntk_std_scale + y_pred
            prob = F_samples.softmax(-1).mean(0)

            probs.append(prob)
            targets.append(y)

            print("done with batch ", len(probs))
            print(time.time() - start)

        probs, targets = torch.cat(probs), torch.cat(targets)
        confidences, predictions = torch.max(probs, 1)

        acc = (predictions == targets).float().mean().item()
        ece = _ECELoss()(confidences, predictions, targets).item()
        ece_laplace = ECE(bins=15).measure(probs.cpu().numpy(), targets.cpu().numpy())
        loss = F.cross_entropy(probs.log(), targets).item()

    print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))
    print(ece_laplace)


def test(test_loader, model, device, args):
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            yy = torch.zeros(20, y_batch.shape[0], 10).to(device)
            for i in range(20):
                y_pred = model(x_batch)
                yy[i] = y_pred
                print(yy.mean(0))
            
            y_pred = yy.mean(0)
#             y_pred = model(x_batch)

            preds.append(y_pred)
            targets.append(y_batch)

        preds, targets = torch.cat(preds), torch.cat(targets)
        probs = preds.softmax(-1)
        confidences, predictions = torch.max(probs, 1)

        acc = (predictions == targets).float().mean().item()
        ece = _ECELoss()(confidences, predictions, targets).item()
        loss = F.cross_entropy(preds, targets).item()

    print("Test results: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}".format(loss, acc, ece))

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

def cifar_loaders(args, batch_size=None, noaug=None):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
        data_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()

        if noaug:
            T = transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        else:
            T = transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=args.data_root, train=True, transform=T, download=True),
            batch_size=batch_size or args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=args.data_root, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[x / 255 for x in [129.3, 124.1, 112.4]],
                                         std=[x / 255 for x in [68.2, 65.4, 70.4]])
        data_mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]]).view(1,-1,1,1).cuda()
        data_std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]]).view(1,-1,1,1).cuda()

        if noaug:
            T = transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        else:
            T = transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=args.data_root, train=True, transform=T, download=True),
            batch_size=batch_size or args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=args.data_root, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'test_cifar10':
        # Data loading code
        dataset_name = config.DATASET.DATASET
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        # For reference: classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)
        gpus = list(config.GPUS)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size or config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size or config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )
    else:
        raise NotImplementedError

    return train_loader, val_loader

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
        L = torch.cholesky(A, upper=upper, out=out)
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
        for i in range(5):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e

if __name__ == '__main__':
    main()