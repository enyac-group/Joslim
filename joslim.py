import os
import sys
import copy
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.drivers import test, get_dataloader

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels import GridInterpolationKernel, AdditiveStructureKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils import standardize

import model as models

from math import cos, pi
from torch.utils.tensorboard import SummaryWriter

import PIL

from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch import distributed as dist

from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors
from torch._utils import _take_tensors
from collections import OrderedDict

writer = None

models = models.__dict__

def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def allreduce_grads(model, world_size, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1).mean()
        return loss

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = F.softmax(target,dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob).mean()
        return cross_entropy_loss

def set_lr(optim, lr):
    for params_group in optim.param_groups:
        params_group['lr'] = lr

def calculate_lr(initlr, cur_step, total_steps, warmup_steps):
    if cur_step < warmup_steps:
        curr_lr = initlr * (cur_step / warmup_steps)
    else:
        if args.scheduler == 'cosine_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1 + cos(pi * T / (N-1))) / 2
        elif args.scheduler == 'linear_decay':
            N = (total_steps-warmup_steps)
            T = (cur_step - warmup_steps)
            curr_lr = initlr * (1-(float(T)/N))
    return curr_lr


class RandAcquisition(AcquisitionFunction):
    def setup(self, obj1, obj2, multiplier=None):
        self.obj1 = obj1
        self.obj2 = obj2
        self.rand = torch.rand(1) if multiplier is None else multiplier

    def forward(self, X):
        linear_weighted_sum = (1-self.rand) * (self.obj1(X)-args.baseline) + self.rand * (self.obj2(X)-args.baseline)
        # NOTE: This is just the augmented Tchebyshev scalarization (c.f. equatino 9 of https://arxiv.org/pdf/1805.12168.pdf)
        return -1*(torch.max((1-self.rand) * (self.obj1(X)-args.baseline), self.rand * (self.obj2(X)-args.baseline)) + (1e-6 * linear_weighted_sum))


def is_pareto_efficient(costs, return_mask = True, epsilon=0):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    # NOTE: This is the non-dominated sorting
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
        self.batch_size = batch_size
    
        if 'CIFAR100' in dataset:
            num_classes = 100
            self.img_size = 32
        elif 'CIFAR10' in dataset:
            num_classes = 10
            self.img_size = 32
        elif 'ImageNet' in dataset:
            num_classes = 1000
            self.img_size = 224

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, eval(args.interpolation), True, args.slim_dataaug, args.scale_ratio, num_gpus=args.world_size, datasize=args.datasize)

        self.dummy = torch.ones(1,3,self.img_size,self.img_size).to(device)

        self.num_classes = num_classes
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        self.sample_pool = None
        if sample_pool is not None:
            self.sample_pool = torch.load(sample_pool)['X']

        self.sampling_weights = np.ones(50)
        if hasattr(model, 'module'):
            self.search_dim = model.module.search_dim
        else:
            self.search_dim = model.search_dim

    def sample_arch(self, START_BO, g, steps, hdim, og_flops, full_val_loss, target_flops=0):
        if args.slim:
            if self.sample_pool is not None:
                idx = np.random.choice(len(self.sample_pool), 1)[0]
                parameterization = self.sample_pool[idx]
            else:
                if target_flops == 0:
                    parameterization = np.random.uniform(args.lower_channel, args.upper_channel, hdim)
                else:
                    parameterization = np.ones(hdim) * args.lower_channel

        else:
            if g < START_BO:
                if self.sample_pool is not None:
                    idx = np.random.choice(len(self.sample_pool), 1)[0]
                    parameterization = self.sample_pool[idx]
                else:
                    if target_flops == 0:
                        f = np.random.rand(1) * (args.upper_channel-args.lower_channel) + args.lower_channel
                    else:
                        f = args.lower_channel
                    parameterization = np.ones(hdim) * f
            elif g == START_BO:
                if target_flops == 0:
                    parameterization = np.ones(hdim)
                else:
                    f = args.lower_channel
                    parameterization = np.ones(hdim) * f
            else:
                rand = torch.rand(1).cuda(self.device)

                train_X = torch.FloatTensor(self.X).cuda(self.device)
                train_Y_loss = torch.FloatTensor(np.array(self.Y)[:, 0].reshape(-1, 1)).cuda(self.device)
                train_Y_loss = standardize(train_Y_loss)

                train_Y_cost = torch.FloatTensor(np.array(self.Y)[:, 1].reshape(-1, 1)).cuda(self.device)
                train_Y_cost = standardize(train_Y_cost)

                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                        num_dims=train_X.shape[1]
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                )

                new_train_X = train_X
                gp_loss = SingleTaskGP(new_train_X, train_Y_loss, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(gp_loss.likelihood, gp_loss)
                mll = mll.to(self.device)
                fit_gpytorch_model(mll)

                # Use add-gp for cost
                covar_module = AdditiveStructureKernel(
                    ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            lengthscale_prior=GammaPrior(3.0, 6.0),
                            num_dims=1
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    ),
                    num_dims=train_X.shape[1]
                )
                gp_cost = SingleTaskGP(new_train_X, train_Y_cost, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(gp_cost.likelihood, gp_cost)
                mll = mll.to(self.device)
                fit_gpytorch_model(mll)

                UCB_loss = UpperConfidenceBound(gp_loss, beta=args.beta).cuda(self.device)
                UCB_cost = UpperConfidenceBound(gp_cost, beta=args.beta).cuda(self.device)
                self.mobo_obj = RandAcquisition(UCB_loss).cuda(self.device)
                self.mobo_obj.setup(UCB_loss, UCB_cost, rand)

                lower = torch.ones(new_train_X.shape[1])*args.lower_channel
                upper = torch.ones(new_train_X.shape[1])*args.upper_channel
                self.mobo_bounds = torch.stack([lower, upper]).cuda(self.device)

                # NOTE: uniformly sample FLOPs
                val = np.linspace(args.lower_flops, 1, 50)
                chosen_target_flops = np.random.choice(val, p=(self.sampling_weights/np.sum(self.sampling_weights)))
                
                lower_bnd, upper_bnd = 0, 1
                lmda = 0.5
                for i in range(10):
                    self.mobo_obj.rand = lmda

                    parameterization, acq_value = optimize_acqf(
                        self.mobo_obj, bounds=self.mobo_bounds, q=1, num_restarts=5, raw_samples=1000,
                    )

                    parameterization = parameterization[0].cpu().numpy()

                    parameterization = np.clip(parameterization, args.lower_channel, args.upper_channel)

                    if hasattr(self.model, 'module'):
                        sim_flops = self.model.module.get_flops_from_wm(parameterization)
                    else:
                        sim_flops = self.model.get_flops_from_wm(parameterization)
                    ratio = sim_flops/og_flops

                    if np.abs(ratio - chosen_target_flops) <= 0.02:
                        break
                    if args.baseline > 0:
                        if ratio < chosen_target_flops:
                            lower_bnd = lmda
                            lmda = (lmda + upper_bnd) / 2
                        elif ratio > chosen_target_flops:
                            upper_bnd = lmda
                            lmda = (lmda + lower_bnd) / 2
                    else:
                        if ratio < chosen_target_flops:
                            upper_bnd = lmda
                            lmda = (lmda + lower_bnd) / 2
                        elif ratio > chosen_target_flops:
                            lower_bnd = lmda
                            lmda = (lmda + upper_bnd) / 2
                rand[0] = lmda
                writer.add_scalar('Binary search trials', i, steps)

        return parameterization, self.sampling_weights/np.sum(self.sampling_weights)

    def train(self, args):
        START_BO = args.prior_points
        self.population_data = []

        # Optimizer
        iters_per_epoch = len(self.train_loader)
        ### all parameter ####
        no_wd_params, wd_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if ".bn" in name or '.bias' in name:
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
        no_wd_params = nn.ParameterList(no_wd_params)
        wd_params = nn.ParameterList(wd_params)
        lr = args.baselr * (args.batch_size / 256.)

        if args.warmup > 0:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr/float(iters_per_epoch*args.warmup), momentum=args.mmt, nesterov=args.nesterov)
        else:
            optimizer = torch.optim.SGD([
                            {'params': no_wd_params, 'weight_decay':0.},
                            {'params': wd_params, 'weight_decay': args.wd},
                        ], lr, momentum=args.mmt, nesterov=args.nesterov)
        lrinfo = {'initlr': lr, 'warmup_steps': args.warmup*iters_per_epoch,
                'total_steps': args.epochs*iters_per_epoch}

        criterion = CrossEntropyLabelSmooth(self.num_classes, args.label_smoothing).to(self.device)
        kd = CrossEntropyLossSoft().cuda(self.device)

        self.model.eval()
        o = self.model(self.dummy)
        # parameterization is layer-wise width multipliers
        parameterization = np.ones(self.search_dim)
        if hasattr(self.model, 'module'):
            og_flops = self.model.module.get_flops_from_wm(parameterization)
        else:
            og_flops = self.model.get_flops_from_wm(parameterization)

        if args.lower_channel != 0:
            parameterization = np.ones(self.search_dim) * args.lower_channel
            if hasattr(self.model, 'module'):
                sim_flops = self.model.module.get_flops_from_wm(parameterization)
            else:
                sim_flops = self.model.get_flops_from_wm(parameterization)
            args.lower_flops = (float(sim_flops) / og_flops)
            if args.local_rank == 0:
                print('Lower flops based on lower channel: {}'.format(args.lower_flops))
        if args.local_rank == 0:
            print('Full MFLOPs: {:.3f}'.format(og_flops/1e6))


        self.X = None
        self.Y = []

        g = 0
        start_epoch = 0
        maxloss = 0
        minloss = 0
        ratio_visited = []
        archs = []

        if os.path.exists(os.path.join('./checkpoint/', '{}.pt'.format(args.name))):
            ckpt = torch.load(os.path.join('./checkpoint/', '{}.pt'.format(args.name)))
            self.X = ckpt['X']
            self.Y = ckpt['Y']
            self.population_data = ckpt['population_data']
            self.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optim_state_dict'])
            start_epoch = ckpt['epoch']+1
            if len(self.population_data) > 1:
                g = len(self.X)
                archs = [data['filters'] for data in self.population_data[-args.num_sampled_arch:]]
            if 'ratio_visited' in ckpt:
                ratio_visited = ckpt['ratio_visited']
            if args.local_rank == 0:
                print('Loading checkpoint from epoch {}'.format(start_epoch-1))


        full_val_loss = 0
        val_iter = iter(self.val_loader)

        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            start_time = time.time()
            for i, (batch, label) in enumerate(self.train_loader):
                self.model.train()
                cur_step = iters_per_epoch*epoch+i
                lr = calculate_lr(lrinfo['initlr'], cur_step, lrinfo['total_steps'], lrinfo['warmup_steps'])
                set_lr(optimizer, lr)
                batch, label = batch.to(self.device), label.to(self.device)

                if not args.normal_training:
                    if not args.slim:
                        if cur_step % args.tau == 0:
                            # NOTE: Calibration of historical data
                            if len(self.Y) > 1:
                                diff = 0
                                try:
                                    val_batch, val_label = next(val_iter)
                                except:
                                    val_iter = iter(self.val_loader)
                                    val_batch, val_label = next(val_iter)
                                val_batch, val_label = val_batch.to(self.device), val_label.to(self.device)
                                for j in range(len(self.Y)):
                                    with torch.no_grad():
                                        if hasattr(self.model, 'module'):
                                            self.model.module.set_real_ch(self.population_data[j]['filters'])
                                        else:
                                            self.model.set_real_ch(self.population_data[j]['filters'])
                                        output = self.model(val_batch)
                                        loss = criterion(output, val_label).item()

                                        if self.Y[j][1] == 1:
                                            full_val_loss = loss

                                        diff += np.abs(loss - self.Y[j][0])
                                        self.Y[j][0] = loss
                                        self.population_data[j]['loss'] = loss

                    if cur_step % args.tau == 0:
                        archs = []
                        ratios = []
                        sampled_sim_flops = []
                        parameterizations = []
                        # Sample architecture
                        for _ in range(args.num_sampled_arch):
                            parameterization, weights = self.sample_arch(START_BO, g, cur_step, self.search_dim, og_flops, full_val_loss)
                            if not args.slim:
                                if hasattr(self.model, 'module'):
                                    sim_flops = self.model.module.get_flops_from_wm(parameterization)
                                else:
                                    sim_flops = self.model.get_flops_from_wm(parameterization)
                                sampled_sim_flops.append(sim_flops)
                                ratio = sim_flops/og_flops
                                ratios.append(ratio)
                                ratio_visited.append(ratio)

                                parameterizations.append(parameterization)
                                g += 1

                            if hasattr(self.model, 'module'):
                                archs.append(self.model.module.decode_wm(parameterization))
                            else:
                                archs.append(self.model.decode_wm(parameterization))

                        if not args.slim:
                            if self.X is None:
                                self.X = np.array(parameterizations)
                            else:
                                self.X = np.concatenate([self.X, parameterizations], axis=0)
                            for ratio, sim_flops, filters in zip(ratios, sampled_sim_flops, archs):
                                self.Y.append([0, ratio])
                                self.population_data.append({'loss': 0, 'flops': sim_flops, 'ratio': ratio, 'filters': filters})


                        # Smallest model
                        parameterization = np.ones(self.search_dim) * args.lower_channel
                        if hasattr(self.model, 'module'):
                            filters = self.model.module.decode_wm(parameterization)
                        else:
                            filters = self.model.decode_wm(parameterization)
                        archs.append(filters)

                # Inplace distillation
                self.model.zero_grad()

                if hasattr(self.model, 'module'):
                    filters = self.model.module.decode_wm(np.ones(self.search_dim))
                    self.model.module.set_real_ch(filters)
                else:
                    filters = self.model.decode_wm(np.ones(self.search_dim))
                    self.model.set_real_ch(filters)
                t_output = self.model(batch)
                loss = criterion(t_output, label)
                loss.backward()

                if args.distributed:
                    maxloss = reduce_tensor(loss.data, args.world_size).item()
                else:
                    maxloss = loss.item()

                for filters in archs:
                    if hasattr(self.model, 'module'):
                        self.model.module.set_real_ch(filters)
                    else:
                        self.model.set_real_ch(filters)
                    output = self.model(batch)
                    loss = kd(output, t_output.detach())
                    loss.backward()

                    if args.distributed:
                        minloss = reduce_tensor(loss.data, args.world_size).item()
                    else:
                        minloss = loss.item()

                if cur_step % args.print_freq == 0 and args.local_rank == 0:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    writer.add_scalar('Loss for largest model', maxloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Loss for smallest model', minloss, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Learning rate', lr, epoch*len(self.train_loader)+i)
                    print('Batch {}/{} | SuperLoss: {:.3f}, MinLoss: {:.3f}, LR: {:.4f}'.format(i, len(self.train_loader), maxloss, minloss, lr))

                if args.distributed:
                    allreduce_grads(model, args.world_size)
                optimizer.step()
                sys.stdout.flush()

            if not os.path.exists('./checkpoint/') and args.local_rank == 0:
                os.makedirs('./checkpoint/')
            if args.local_rank == 0:
                torch.save({'model_state_dict': self.model.state_dict(), 'optim_state_dict': optimizer.state_dict(),
                            'epoch': epoch, 'population_data': self.population_data, 'X': self.X, 'Y': self.Y, 'ratio_visited': ratio_visited}, os.path.join('./checkpoint/', '{}.pt'.format(args.name)))
            if len(ratio_visited) > 0 and args.local_rank == 0:
                writer.add_histogram('FLOPs visited', np.array(ratio_visited), epoch+1)
            if args.local_rank == 0:
                print('Epoch {} | Time: {:.2f}s'.format(epoch, time.time()-start_time))

            if args.normal_training:
                test_top1, test_top5 = test(self.model, self.test_loader, device=self.device)
                if args.local_rank == 0:
                    writer.add_scalar('Test acc/Top-1', test_top1, epoch+1)
                    writer.add_scalar('Test acc/Top-5', test_top1, epoch+1)


            torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument("--name", type=str, default='test', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='CIFAR100', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument("--network", type=str, default='slim_resnet56', help='The model architecture')
    parser.add_argument("--interpolation", type=str, default='PIL.Image.BILINEAR', help='Image resizing interpolation')
    parser.add_argument("--print_freq", type=int, default=500, help='Logging frequency in iterations')

    # Training
    parser.add_argument("--datasize", type=float, default=1, help='Dataset size ratio')
    parser.add_argument("--epochs", type=int, default=120, help='Number of training epochs')
    parser.add_argument("--warmup", type=int, default=5, help='Number of warmup epochs')
    parser.add_argument("--baselr", type=float, default=0.05, help='The learning rate for fine-tuning')
    parser.add_argument("--scheduler", type=str, default='cosine_decay', help='Support: cosine_decay | linear_decay')
    parser.add_argument("--mmt", type=float, default=0.9, help='Momentum for fine-tuning')
    parser.add_argument("--tau", type=int, default=200, help='training iterations for one architecture')
    parser.add_argument("--wd", type=float, default=1e-4, help='The weight decay used')
    parser.add_argument("--scale_ratio", type=float, default=0.08, help='Scale for random scaling, default: 0.08')
    parser.add_argument("--label_smoothing", type=float, default=1e-1, help='Label smoothing')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training')
    parser.add_argument("--distill", action='store_true', default=False, help='Distillation from pre-trained model')
    parser.add_argument("--normal_training", action='store_true', default=False, help='For independent trained model')
    parser.add_argument("--nesterov", action='store_true', default=False, help='For independent trained model')
    parser.add_argument("--slim_dataaug", action='store_true', default=False, help='Use the data augmentation implemented in universally slimmable network')
    parser.add_argument("--seed", type=int, default=0, help='Random seed')

    # Channel
    parser.add_argument("--lower_channel", type=float, default=0, help='lower bound')
    parser.add_argument("--upper_channel", type=float, default=1, help='upper bound')
    parser.add_argument("--slim", action='store_true', default=False, help='Use slimmable training')
    parser.add_argument("--num_sampled_arch", type=int, default=1, help='Number of arch sampled in between largest and smallest')
    parser.add_argument('--track_flops', nargs='+', default=[0.35, 0.5, 0.75])
    parser.add_argument("--cont_sampling", action='store_true', default=False, help='Continuous sampling previous arch to train')
    parser.add_argument("--sample_pool", type=str, default='none', help='Checkpoint to sample architectures from')
    parser.add_argument("--sync_bn", action='store_true', default=False, help='Use sync bn')

    # GP-related hyper-param (Joslim)
    parser.add_argument("--buffer", type=int, default=1000, help='Buffer for GP')
    parser.add_argument("--beta", type=float, default=0.1, help='For UCB')
    parser.add_argument("--prior_points", type=int, default=10, help='Number of uniform arch for BO')
    parser.add_argument("--baseline", type=int, default=5, help='Use for scalarization')

    # Distributed
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed and args.rank == 0:
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    elif args.rank == 0:
        print('Training with a single process on 1 GPU.')

    if args.local_rank == 0:
        print(args)
    random_seed = 3080 + args.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    if args.local_rank == 0:
        writer = SummaryWriter('./runs/{}'.format(args.name))

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

    if args.distributed:
        model = NativeDDP(model, device_ids=[args.device])

    sample_pool = None if args.sample_pool == 'none' else args.sample_pool
    joslim = Joslim(args.dataset, args.datapath, model, sample_pool, args.batch_size, device=device)

    start = time.time()
    joslim.train(args)
    end = time.time()
    if args.local_rank == 0:
        print('Total time: {:.3f}s'.format(end-start))
