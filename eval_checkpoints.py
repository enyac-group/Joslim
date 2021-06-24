import os
import sys
import copy
import time
import math
import torch
import queue
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from utils.drivers import test, get_dataloader

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels import GridInterpolationKernel, AdditiveStructureKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition import UpperConfidenceBound, qMaxValueEntropy
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

import PIL

import model as models

models = models.__dict__

def is_pareto_efficient(costs, return_mask = True, epsilon=0):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index]-epsilon, axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

class Joslim:
    def __init__(self, dataset, datapath, model, sample_pool=None, batch_size=32, device='cuda'):
        self.device = device
        self.img_size = 32 if 'CIFAR' in dataset else 224
        self.dummy = torch.ones(1,3,self.img_size,self.img_size).to(device)
        self.batch_size = batch_size
    
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, eval(args.interpolation), True, args.slim_dataaug, args.scale_ratio)

        if 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10
        elif 'ImageNet' in dataset:
            num_classes = 1000

        self.num_classes = num_classes
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        self.sample_pool = None
        if sample_pool is not None:
            self.sample_pool = torch.load(sample_pool)['X']

    def eval(self, args):
        data_dict = torch.load(os.path.join('./checkpoint', '{}.pt'.format(args.name)))

        population_data = data_dict['population_data']

        self.model.load_state_dict(data_dict['model_state_dict'])
        if 'epoch' in data_dict:
            print('Load from epoch: {}'.format(data_dict['epoch']))

        self.model.eval()
        o = self.model(self.dummy)
        parameterization = np.ones(self.model.search_dim)
        og_flops = self.model.get_flops_from_wm(parameterization)

        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            full_real_ch = self.model.decode_wm(np.ones(self.model.search_dim))
            self.model.set_real_ch(full_real_ch)
            full_loss = 0
            for i, (batch, label) in enumerate(self.train_loader):
                batch = batch.to('cuda')
                label = label.to('cuda')
                out = self.model(batch)
                full_loss += criterion(out, label).item()
                if 'CIFAR' not in args.dataset and i == 1:
                    break

            full_test_top1, full_test_top5  = test(self.model, self.test_loader, device='cuda')

        print('Full: {:.2f}, {:.2f} MFLOPS: {:.3f}'.format(full_test_top1, full_test_top5, og_flops*1e-6))


        print('Alpha dim: {}'.format(self.model.search_dim))

        if args.lower_channel != 0:
            parameterization = np.ones(self.model.search_dim) * args.lower_channel
            smallest_flops = self.model.get_flops_from_wm(parameterization)
            args.lower_flops = (float(smallest_flops) / og_flops)
            print('Lower flops based on lower channel: {}'.format(args.lower_flops))

        with torch.no_grad():
            f = args.lower_channel
            parameterization = np.ones(self.model.search_dim) * f
            filters = self.model.decode_wm(parameterization)
            self.model.set_real_ch(filters)

            # Eval on validation set
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_running_stats()
                    m.momentum = None

            loss = 0
            for i, (batch, label) in enumerate(self.train_loader):
                batch = batch.to('cuda')
                label = label.to('cuda')
                out = self.model(batch)
                loss += criterion(out, label).item()
                if 'CIFAR' not in args.dataset and i == 1:
                    break

            smallest_test_top1, smallest_test_top5  = test(self.model, self.test_loader, device='cuda')

        print('Smallest: {:.2f}, {:.2f} MFLOPS: {:.3f}'.format(smallest_test_top1, smallest_test_top5, smallest_flops*1e-6))

        test_top1s = [smallest_test_top1]
        test_top5s = [smallest_test_top5]
        new_flops = [float(smallest_flops)/og_flops]

        if not args.uniform and self.sample_pool is None:
            costs = []
            filters = []
            for i in range(len(population_data)-2):
                if 'acc' in population_data[i]:
                    costs.append([1-population_data[i]['acc']/100., population_data[i]['ratio']])
                elif 'loss' in population_data[i]:
                    costs.append([population_data[i]['loss'], population_data[i]['ratio']])
                filters.append(population_data[i]['filters'])

            max_flops = 0
            costs = np.array(costs)
            tmp_costs = np.array(costs)
            global_efficient_mask = np.zeros(len(costs))
            while max_flops < 0.9:
                efficient_mask = is_pareto_efficient(tmp_costs)
                efficient_mask[costs[:, 1] < max_flops] = False
                if np.sum(efficient_mask) == 0:
                    break
                max_flops = np.max(costs[efficient_mask][:, 1])
                global_efficient_mask = np.logical_or(global_efficient_mask, efficient_mask)
                tmp_costs[efficient_mask] = np.ones_like(tmp_costs[efficient_mask])
            efficient_mask = global_efficient_mask 
            filters = np.array(filters)[efficient_mask]
            ratios = costs[efficient_mask][:, 1]

            out_X = data_dict['X'][:-2, :]
            out_X = out_X[efficient_mask]

            torch.save({'X': out_X}, os.path.join('./checkpoint', '{}_sample_pool.pt'.format(args.name)))

            index_set = np.arange(len(filters))

            for li in range(len(index_set)):
                real_ch = filters[index_set[li]]
                ratio = ratios[li]
                self.model.set_real_ch(real_ch)

                # Eval on validation set
                loss = 0
                with torch.no_grad():
                    self.model.train()
                    for m in self.model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.reset_running_stats()
                            m.momentum = None
                    for i, (batch, label) in enumerate(self.train_loader):
                        batch = batch.to('cuda')
                        label = label.to('cuda')
                        out = self.model(batch)
                        loss += criterion(out, label).item()
                        if 'CIFAR' not in args.dataset and i == 1:
                            break
                    test_top1, test_top5 = test(self.model, self.test_loader, device='cuda')
                test_top1s.append(test_top1)
                test_top5s.append(test_top5)
                new_flops.append(ratio)

                print('({}/{}) Acc: {:.2f} {:.2f}, MFLOPs: {:.3f} ({:.2f} %)'.format(li, len(filters), test_top1, test_top5, og_flops*ratio*1e-6, ratio*100.))

        elif self.sample_pool is not None:
            for j in range(len(self.sample_pool)):
                wm = self.sample_pool[j]
                self.model.set_real_ch(self.model.decode_wm(wm))
                flops = self.model.get_flops_from_wm(wm)

                # Eval on validation set
                self.model.train()
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = None
                loss = 0

                with torch.no_grad():
                    self.model.train()
                    for i, (batch, label) in enumerate(self.train_loader):
                        batch = batch.to('cuda')
                        label = label.to('cuda')
                        out = self.model(batch)
                        loss += criterion(out, label).item()
                        if 'CIFAR' not in args.dataset and i == 1:
                            break
                    test_top1, test_top5 = test(self.model, self.test_loader, device='cuda')
                test_top1s.append(test_top1)
                test_top5s.append(test_top5)
                new_flops.append(flops/og_flops)

                print('Acc: {:.2f} {:.2f}, MFLOPs: {:.3f} ({:.2f} %)'.format(test_top1, test_top5, flops*1e-6, flops/og_flops*100.))
        else:
            num = 40
            index_set = np.sqrt((np.arange(num) / num)*(args.upper_flops-args.lower_flops) + args.lower_flops)
            for f in index_set:
                parameterization = np.ones(self.model.search_dim) * f
                flops = self.model.get_flops_from_wm(parameterization)
                filters = self.model.decode_wm(parameterization)
                self.model.set_real_ch(filters)

                # Eval on validation set
                self.model.train()
                for m in selfmodel.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats()
                        m.momentum = None

                loss = 0
                with torch.no_grad():
                    selfmodel.train()
                    for i, (batch, label) in enumerate(self.train_loader):
                        batch = batch.to('cuda')
                        label = label.to('cuda')
                        out = selfmodel(batch)
                        loss += criterion(out, label).item()
                        if 'CIFAR' not in args.dataset and i == 1:
                            break
                    test_top1, test_top5 = test(selfmodel, self.test_loader, device='cuda')
                test_top1s.append(test_top1)
                test_top5s.append(test_top5)
                new_flops.append(ratio)

                print('WM: {:.2f} Acc: {:.2f} {:.2f}, MFLOPs: {:.3f} ({:.2f} %)'.format(f, test_top1, test_top5, flops*1e-6, flops/og_flops*100.))


        efficient_mask = is_pareto_efficient(np.stack([100-np.array(test_top1s), np.array(new_flops)], axis=1))
        idx = np.where(efficient_mask != 0)[0]
        new_flops = np.array(new_flops)[efficient_mask]
        test_top1s = np.array(test_top1s)[efficient_mask]
        test_top5s = np.array(test_top5s)[efficient_mask]

        new_flops = np.concatenate([new_flops.reshape(-1), [1]])
        test_top1s = np.concatenate([test_top1s.reshape(-1), [full_test_top1]])
        test_top5s = np.concatenate([test_top5s.reshape(-1), [full_test_top5]])

        sorted_idx = np.argsort(new_flops)
        new_flops = new_flops[sorted_idx]
        test_top1s = test_top1s[sorted_idx]
        test_top5s = test_top5s[sorted_idx]

        if not os.path.exists('./results/') and args.local_rank == 0:
            os.makedirs('./results/')
        if not args.uniform:
            np.savetxt('results/{}_eval_pareto.txt'.format(args.name), np.stack([new_flops, test_top1s, test_top5s]))
        else:
            np.savetxt('results/{}_eval_uniform_pareto.txt'.format(args.name), np.stack([new_flops, test_top1s, test_top5s]))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='ImageNet', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--logging", action='store_true', default=False, help='Log the output')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training.')
    parser.add_argument("--uniform", action='store_true', default=False, help='Use Adam instead of SGD')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--network", type=str, default='slim_mobilenetv2', help='The model architecture')
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')
    parser.add_argument("--sample_pool", type=str, default='none', help='Checkpoint to sample architectures from')
    parser.add_argument("--scale_ratio", type=float, default=0.08, help='Scale for random scaling, default: 0.08')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.logging:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open('log/{}.log'.format(args.name), 'w')
    print(args)

    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageNet' in args.dataset:
        num_classes = 1000

    device = torch.cuda.current_device()

    assert args.network in models
    model = models[args.network](num_classes=num_classes)
    model = model.to(device)

    sample_pool = None if args.sample_pool == 'none' else args.sample_pool
    joslim = Joslim(args.dataset, args.datapath, model, sample_pool, args.batch_size, device=device)

    joslim.eval(args)