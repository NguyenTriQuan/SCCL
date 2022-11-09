
from pyexpat import features
from re import S
from tkinter.tix import InputOnly
from traceback import print_tb
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal, Normal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, device, isin, seed
from typing import Optional, Any
from torch.nn import init
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utils import *
from typing import Optional, List, Tuple, Union
import sys
from arguments import get_args
args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.in_features = 0
        self.out_features = 0
        self.p = args.ensemble_drop
            
        self.weight = nn.ParameterList([])
        self.fwt_weight = nn.ParameterList([])
        self.bwt_weight = nn.ParameterList([])
        if bias:
            self.bias = nn.ParameterList([])
        else:
            self.bias = None

        self.shape_out = [self.out_features]
        self.shape_in = [self.in_features]

        self.norm_type = norm_type

        if norm_type:
            self.norm_layer = DynamicNorm(self.out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        self.next_layers = next_layers # where output of this layer go
        self.s = s
        self.mask = None
        
        self.cur_task = -1
        self.track = False
        self.out_tracked = None

    def forward(self, x, t): 
        if self.out_tracked is not None:
            return self.out_tracked

        weight, bias = self.get_parameters(t)

        if weight.numel() == 0:
            return None

        if isinstance(self, DynamicLinear):
            x = F.linear(x, weight, bias)
            view = (1, -1)
        else:
            x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            view = (1, -1, 1, 1)

        if self.norm_layer is not None:
            x = self.norm_layer(x, t)

        if args.prune_method == 'bs':
            if self.mask is not None:
                x *= self.mask.view(view)
            elif self.track:
                self.out_tracked = x
        return x

    def norm_in(self):
        weight = torch.cat([self.fwt_weight[-1], self.weight[-1]], dim=1)
        norm = weight.norm(2, dim=self.dim_in)
        if self.bias is not None:
            norm = (norm ** 2 + self.bias[-1][self.shape_out[-2]:] ** 2) ** 0.5
        return norm

    def norm_out(self, n=0):
        weight = torch.cat([self.next_layers[n].bwt_weight[-1], self.next_layers[n].weight[-1]], dim=0)
        if isinstance(self, DynamicConv2D) and isinstance(self.next_layers[0], DynamicLinear):
            weight = weight.view(self.next_layers[n].weight[-1].shape[0] + self.next_layers[n].bwt_weight[-1].shape[0], 
                                self.weight[-1].shape[0], self.next_layers[n].s, self.next_layers[n].s)
        return weight.norm(2, dim=self.dim_out)

    def get_optim_params(self):
        params = []
        params += [self.weight[-1], self.fwt_weight[-1], self.bwt_weight[-1]]
        if self.bias is not None:
            params += [self.bias[-1]]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight[i].numel() + self.fwt_weight[i].numel() + self.bwt_weight[i].numel()
            if self.bias:
                count += self.bias[i].numel()
            if self.norm_layer:
                if self.norm_layer.affine:
                    count += self.norm_layer.weight[i].numel() + self.norm_layer.bias[i].numel()
        return count

    def get_parameters(self, t):
        if self.last_layer:
            weight = torch.cat([self.fwt_weight[t], self.weight[t]], dim=1)
            bias = self.bias[t][self.shape_out[t]:self.shape_out[t+1]] if self.bias is not None else None
            return weight, bias

        weight = torch.empty(0).to(device)

        for i in range(t):
            weight = torch.cat([torch.cat([weight, self.bwt_weight[i]], dim=1), 
                                torch.cat([self.fwt_weight[i], self.weight[i]], dim=1)], dim=0)

        # print(self.weight[t].shape, self.fwt_weight[t].shape, self.bwt_weight[t].shape)
        weight = F.dropout(weight, self.p, self.training)
        weight = torch.cat([torch.cat([weight, self.bwt_weight[t]], dim=1),
                            torch.cat([self.fwt_weight[t], self.weight[t]], dim=1)], dim=0)

        bias = self.bias[t] if self.bias is not None else None
        # weight = F.dropout(weight, self.p, self.training)
        return weight, bias

    def squeeze(self, optim_state):
        def apply_mask_out(param, mask_out):
            param.data = param.data[mask_out].clone()
            param.grad = None
            param_states = optim_state[param]
            for name, state in param_states.items():
                if isinstance(state, torch.Tensor):
                    param_states[name] = state[mask_out].clone()

        def apply_mask_in(param, mask_in):
            param.data = param.data[:, mask_in].clone()
            param.grad = None
            param_states = optim_state[param]
            for name, state in param_states.items():
                if isinstance(state, torch.Tensor):
                    param_states[name] = state[:, mask_in].clone()


        if self.mask is not None:
            if args.prune_method == 'pgd':
                mask_out = self.mask
            else:
                mask_out = self.mask[self.shape_out[-2]:]
            apply_mask_out(self.weight[-1], mask_out)
            apply_mask_out(self.fwt_weight[-1], mask_out)

            self.out_features = self.shape_out[-2] + self.weight[-1].shape[0]
            self.shape_out[-1] = self.out_features

            if args.prune_method == 'pgd':
                mask = torch.cat([torch.ones(self.shape_out[-2], dtype=bool, device=device), self.mask])
            else:
                mask = self.mask

            if self.bias is not None:
                apply_mask_out(self.bias[-1], mask)

            if self.norm_layer:
                if self.norm_layer.affine:
                    apply_mask_out(self.norm_layer.weight[-1], mask)
                    apply_mask_out(self.norm_layer.bias[-1], mask)

                if self.norm_layer.track_running_stats:
                    self.norm_layer.running_mean[-1] = self.norm_layer.running_mean[-1][mask]
                    self.norm_layer.running_var[-1] = self.norm_layer.running_var[-1][mask]

                self.norm_layer.num_features = self.out_features
                self.norm_layer.shape[-1] = self.out_features

            for m in self.next_layers:
                if args.prune_method == 'pgd':
                    mask_in = self.mask
                else:
                    mask_in = self.mask[self.shape_out[-2]:]
                if isinstance(m, DynamicLinear) and isinstance(self, DynamicConv2D):
                    mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0),m.s,m.s).contiguous().view(-1)

                apply_mask_in(m.weight[-1], mask_in)
                apply_mask_in(m.bwt_weight[-1], mask_in)

                m.in_features = m.shape_in[-2] + m.weight[-1].shape[1]
                m.shape_in[-1] = m.in_features
  
            self.mask = None
            if 'adaptreg' not in args.ablation:
                self.get_reg_strength()

    def get_reg_strength(self):
        self.strength_in = self.weight[-1].numel() + self.fwt_weight[-1].numel()
        self.strength_out = [0]
        for m in self.next_layers:
            strength_out = m.weight[-1].numel() + m.bwt_weight[-1].numel()
            self.strength_out.append(strength_out)
        self.strength_out = sum(self.strength_out)

        self.strength = (self.strength_in + self.strength_out)

    def expand(self, add_in=None, add_out=None, ablation='full'):
        self.cur_task += 1
        if add_in is None:
            if args.fix:
                add_in = self.base_in_features - self.in_features
            else:
                add_in = self.base_in_features
        if add_out is None:
            if args.fix:
                add_out = self.base_out_features - self.out_features
            else:
                add_out = self.base_out_features

        if self.first_layer:
            if self.cur_task == 0:
                add_in = self.base_in_features
            else:
                add_in = 0

        gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        if isinstance(self, DynamicLinear):
            fan_in = self.in_features + add_in
            bound_std = gain / math.sqrt(fan_in)
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).normal_(0, bound_std).to(device)))
            # old neurons to new neurons
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features).normal_(0, bound_std).to(device)))
            # new neurons to old neurons
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in).normal_(0, bound_std).to(device)))
        else:
            fan_in = (self.in_features + add_in) * np.prod(self.kernel_size)
            bound_std = gain / math.sqrt(fan_in)
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            # old neurons to new neurons
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))
            # new neurons to old neurons
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in // self.groups, *self.kernel_size).normal_(0, bound_std).to(device)))

        # print(self.weight[-1].shape, self.fwt_weight[-1].shape, self.bwt_weight[-1].shape)
        self.in_features += add_in
        self.out_features += add_out

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)
        # print(self.shape_in, self.shape_out)

        if self.bias is not None:
            self.bias.append(nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0).to(device)))

        if self.norm_layer:
            self.norm_layer.expand(add_out)                
        
        self.mask = None
        if self.cur_task > 0:
            self.weight[-2].requires_grad = False
            self.fwt_weight[-2].requires_grad = False
            self.bwt_weight[-2].requires_grad = False
            if self.bias is not None:
                self.bias[-2].requires_grad = False
            if self.norm_layer:
                if self.norm_layer.affine:
                    self.norm_layer.weight[-2].requires_grad = False
                    self.norm_layer.bias[-2].requires_grad = False

    def get_reg(self):
        reg = 0
        reg += self.norm_in().sum() * self.strength_in
        for n in range(len(self.next_layers)):
            reg += self.norm_out(n).sum() * self.strength_out
            
        if self.norm_layer:
            if self.norm_layer.affine:
                reg += self.norm_layer.norm().sum() * self.strength

        return reg

    def get_importance(self):
        norm = self.norm_in() 
        if len(self.next_layers) != 0:
            norm_out = 0
            for n in range(len(self.next_layers)):
                norm_out += self.norm_out(n)
            norm *= norm_out
        if self.norm_layer:
            if self.norm_layer.affine:
                norm *= self.norm_layer.norm()

        return norm
    
    def proximal_gradient_descent(self, lr, lamb, total_strength):

        with torch.no_grad():
            if 'strength' not in args.ablation:
                strength_in = self.strength_in/total_strength
                strength_out = self.strength_out/total_strength
                strength = self.strength/total_strength
            else:
                strength_in = 1
                strength_out = 1
                strength = 1
            # group lasso weights in
            norm = self.norm_in()
            aux = 1 - lamb * lr * strength_in / norm
            aux = F.threshold(aux, 0, 0, False)
            self.mask = (aux > 0)

            self.weight[-1].data *= aux.view(self.view_in)
            self.fwt_weight[-1].data *= aux.view(self.view_in)
            if self.bias:
                self.bias[-1].data[self.shape_out[-2]:] *= aux

            if 'outgoing' not in args.ablation:
            # group lasso weights out
                if len(self.next_layers) > 0:
                    mask_temp = False
                    for n, m in enumerate(self.next_layers):
                        norm = self.norm_out(n)
                        aux = 1 - lamb * lr * strength_out / norm
                        aux = F.threshold(aux, 0, 0, False)
                        mask_temp += (aux > 0)
                        if isinstance(m, DynamicLinear) and isinstance(self, DynamicConv2D):
                            aux = aux.view(-1, 1, 1).expand(aux.size(0), m.s, m.s).contiguous().view(-1)
                        m.weight[-1].data *= aux.view(m.view_out)
                        m.bwt_weight[-1].data *= aux.view(m.view_out)                  
                    self.mask *= mask_temp
            # group lasso affine weights
            if 'regaffine' not in args.ablation:
                if self.norm_layer:
                    if self.norm_layer.affine:
                        norm = self.norm_layer.norm()
                        aux = 1 - lamb * lr * strength / norm
                        aux = F.threshold(aux, 0, 0, False)
                        self.mask *= (aux > 0)

                        self.norm_layer.weight[-1].data[self.norm_layer.shape[-2]:] *= aux
                        self.norm_layer.bias[-1].data[self.norm_layer.shape[-2]:] *= aux

class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(DynamicLinear, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

        self.view_in = (-1, 1)
        self.view_out = (1, -1)
        self.dim_in = (1)
        self.dim_out = (0)

            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout):
        super(_DynamicConvNd, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)
        if in_features % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups


class DynamicConv2D(_DynamicConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

        self.view_in = (-1, 1, 1, 1)
        self.view_out = (1, -1, 1, 1)
        self.dim_in = (1, 2, 3)
        self.dim_out = (0, 2, 3)


class DynamicNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True, norm_type=None):
        super(DynamicNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.base_num_features = num_features
        self.num_features = 0
        self.shape = [0]
        self.norm_type = norm_type
        if 'affine' in norm_type:
            self.affine = True
        else:
            self.affine = False

        if 'track' in norm_type:
            self.track_running_stats = True
        else:
            self.track_running_stats = False

        if self.affine:
            self.weight = nn.ParameterList([])
            self.bias = nn.ParameterList([])

        if self.track_running_stats:
            self.running_mean = []
            self.running_var = []
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def expand(self, add_num=None):
        if add_num is None:
            add_num = self.base_num_features

        self.num_features += add_num
        self.shape.append(self.num_features)

        if self.affine:
            self.weight.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(1,1)))
            self.bias.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(0,0)))
            if len(self.weight) > 2:
                self.weight[-2].requires_grad = False
                self.bias[-2].requires_grad = False

        if self.track_running_stats:
            self.running_mean.append(torch.zeros(self.num_features).to(device))
            self.running_var.append(torch.ones(self.num_features).to(device))
            self.num_batches_tracked = 0
    
    def norm(self, t=-1):
        return (self.weight[t][self.shape[t-1]:]**2 + self.bias[t][self.shape[t-1]:]**2) ** 0.5

    def batch_norm(self, input, t=-1):

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked += 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        if len(input.shape) == 4:
            mean = input.mean([0, 2, 3])
            # var = input.var([0, 2, 3], unbiased=False)
            shape = (1, -1, 1, 1)
            var = ((input - mean.view(shape)) ** 2).mean([0, 2, 3])
        else:
            mean = input.mean([0])
            # var = input.var([0], unbiased=False)
            shape = (1, -1)
            var = ((input - mean.view(shape)) ** 2).mean([0])

        # calculate running estimates
        if bn_training:
            if self.track_running_stats:
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean[t] = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean[t]
                    # update running_var with unbiased var
                    self.running_var[t] = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var[t]
        else:
            mean = self.running_mean[t]
            var = self.running_var[t]

        # if 'res' in self.norm_type:
        #     return input / (torch.sqrt(var.view(shape) + self.eps))
        # else:
        #     return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))
        return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))

    def layer_norm(self, input):
        if len(input.shape) == 4:
            mean = input.mean([1, 2, 3])
            var = input.var([1, 2, 3], unbiased=False)
            shape = (-1, 1, 1, 1)
        else:
            mean = input.mean([1])
            var = input.var([1], unbiased=False)
            shape = (-1, 1)

        return (input - mean.view(shape)) / (torch.sqrt(var.view(shape) + self.eps))

    def L2_norm(self, input):
        if len(input.shape) == 4:
            norm = input.norm(2, dim=(1,2,3)).view(-1,1,1,1)
        else:
            norm = input.norm(2, dim=(1)).view(-1,1)

        return input / norm

    def forward(self, input, t=-1):

        # input = self.batch_norm(input, t)
        if 'res' in self.norm_type:
            input = self.batch_norm(input, t) + input
        else:
            input = self.batch_norm(input, t)

        if self.affine:
            weight = self.weight[t]
            bias = self.bias[t]
            if len(input.shape) == 4:
                input = input * weight.view(1,-1,1,1) + bias.view(1,-1,1,1)
            else:
                input = input * weight.view(1,-1) + bias.view(1,-1)

        return input


class re_sigma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, r):
        return sigma.clamp(min=r)

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
            