from asyncio import current_task
from unittest import makeSuite
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal, Normal
import numpy as np
import random
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

def apply_mask_out(param, mask_out, optim_state):
    param.data = param.data[mask_out].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[mask_out].clone()

def apply_mask_in(param, mask_in, optim_state):
    param.data = param.data[:, mask_in].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[:, mask_in].clone()

class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.p = args.ensemble_drop
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.in_features = 0
        self.out_features = 0
            
        self.weight = []
        # bias = False # !!!!!
        self.bias = [] if bias else None

        self.scale = []
        self.shift = []

        self.num_in = []
        self.num_out = []

        self.shape_out = [self.out_features]
        self.shape_in = [self.in_features]

        self.norm_type = norm_type

        if norm_type:
            self.norm_layer = DynamicNorm(self.out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        self.next_layers = next_layers # where output of this layer go
        self.s = s
        self.mask_in = None
        self.mask_out = None
        
        self.cur_task = -1

    def forward(self, x, t, n, m):    
        weight, bias = self.get_parameters(t, n, m)

        if weight.numel() == 0:
            return None
    
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, weight, bias)

        if self.norm_layer is not None:
            output = self.norm_layer(output, m)
        
        return output

    def expand(self, add_in=None, add_out=None, ablation='full'):
        self.cur_task += 1
        if add_in is None:
            add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features

        if self.first_layer:
            if self.cur_task == 0:
                add_in = self.base_in_features
            else:
                add_in = 0

        self.in_features += add_in
        self.out_features += add_out

        self.num_out.append(add_out)
        self.num_in.append(add_in)

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)

        self.weight.append([])
        if isinstance(self, DynamicConv2D):
            # self.gain = torch.nn.init.calculate_gain('relu')
            self.gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            fan_in = self.in_features * np.prod(self.kernel_size)
            bound_std = self.gain / math.sqrt(fan_in)
            for i in range(self.cur_task):
                self.weight[i].append(nn.Parameter(torch.Tensor(self.num_out[-1], self.num_in[i] // self.groups,
                                                            *self.kernel_size).normal_(0, bound_std).to(device)))
                self.weight[-1].append(nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[-1] // self.groups, 
                                                            *self.kernel_size).normal_(0, bound_std).to(device)))
            self.weight[-1].append(nn.Parameter(torch.Tensor(self.num_out[-1], self.num_in[-1] // self.groups, 
                                                            *self.kernel_size).normal_(0, bound_std).to(device)))
        else:
            self.gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            fan_in = self.in_features
            bound_std = self.gain / math.sqrt(fan_in)
            for i in range(self.cur_task):
                self.weight[i].append(nn.Parameter(torch.Tensor(self.num_out[-1], self.num_in[i]).normal_(0, bound_std).to(device)))
                self.weight[-1].append(nn.Parameter(torch.Tensor(self.num_out[i], self.num_in[-1]).normal_(0, bound_std).to(device)))
            self.weight[-1].append(nn.Parameter(torch.Tensor(self.num_out[-1], self.num_in[-1]).normal_(0, bound_std).to(device)))

        # rescale old tasks params
        if self.cur_task > 0:
            self.scale.append([])
            self.shift.append([])
            for i in range(self.cur_task):
                self.scale[-1].append([])
                self.shift[-1].append([])
                for j in range(self.cur_task):
                    if self.weight[i][j].numel() == 0:
                        self.shift[-1][-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
                        self.scale[-1][-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
                    else:
                        if 'shift' in args.ablation:
                            self.shift[-1][-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
                        else:
                            w_mean = self.weight[i][j].mean(dim=self.dim_in).view(self.view_in)
                            self.shift[-1][-1].append(nn.Parameter(w_mean))
                        if 'scale' in args.ablation:
                            self.scale[-1][-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
                        else:
                            w_std = self.weight[i][j].std(dim=self.dim_in, unbiased=False).view(self.view_in)
                            self.scale[-1][-1].append(nn.Parameter(w_std))
                    
            #     if self.cur_task > 1:
            #         for i in range(self.cur_task):
            #             if self.weight[i][-2].numel() == 0:
            #                 self.shift[i].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #                 self.scale[i].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
            #             else:
            #                 w_mean = self.weight[i][-2].mean(dim=self.dim_in).view(self.view_in)
            #                 self.shift[i].append(nn.Parameter(w_mean))
            #                 w_std = self.weight[i][-2].std(dim=self.dim_in, unbiased=False).view(self.view_in)
            #                 self.scale[i].append(nn.Parameter(w_std))
                            
            #             if self.weight[-2][i].numel() == 0:
            #                 self.shift[-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #                 self.scale[-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
            #             else:
            #                 w_mean = self.weight[-2][i].mean(dim=self.dim_in).view(self.view_in)
            #                 self.shift[-1].append(nn.Parameter(w_mean))
            #                 w_std = self.weight[-2][i].std(dim=self.dim_in, unbiased=False).view(self.view_in)
            #                 self.scale[-1].append(nn.Parameter(w_std))
                
            #     if self.weight[-2][-2].numel() == 0:
            #         self.shift[-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #         self.scale[-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
            #     else:
            #         w_mean = self.weight[-2][-2].mean(dim=self.dim_in).view(self.view_in)
            #         self.shift[-1].append(nn.Parameter(w_mean))
            #         w_std = self.weight[-2][-2].std(dim=self.dim_in, unbiased=False).view(self.view_in)
            #         self.scale[-1].append(nn.Parameter(w_std))
            # else:
            #     for i in range(self.cur_task):
            #         self.shift[i].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #         self.shift[-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #         self.scale[i].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
            #         self.scale[-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
            #     self.shift[-1].append(nn.Parameter(torch.zeros(1).to(device), requires_grad=False))
            #     self.scale[-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))

        if self.bias is not None:
            self.bias.append(nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0).to(device)))

        if self.norm_layer:
            self.norm_layer.expand(add_out)             

        # freeze old params
        if self.cur_task > 0:
            self.weight[-2][-2].requires_grad = False
            for i in range(self.cur_task-1):
                self.weight[-2][i].requires_grad = False
                self.weight[i][-2].requires_grad = False
                # for j in range(self.cur_task-1):
                #     self.scale[-2][i][j].requires_grad = False
            if self.bias is not None:
                self.bias[-2].requires_grad = False
            if self.norm_layer:
                if self.norm_layer.affine:
                    self.norm_layer.weight[-2].requires_grad = False
                    self.norm_layer.bias[-2].requires_grad = False

        self.get_reg_strength()

        for i in range(self.cur_task):
            for j in range(self.cur_task):
                print(self.scale[-1][i][j].shape[0], end=' ')
            print()
        print()

    def get_parameters(self, t, n, m):
        if self.last_layer:
            weight = self.weight[n][m]
            bias = self.bias[n][m] if self.bias is not None else None
        elif self.first_layer:
            weight = torch.empty(0).to(device)
            for j in range(m+1):
                weight = torch.cat([weight, self.weight[0][j]], dim=0)
            bias = self.bias[m] if self.bias is not None else None
        else:
            if 'scale' in args.ablation:
                bound_std = 1
            else:
                bound_std = self.gain / math.sqrt(self.shape_in[n+1])
            weight = torch.empty(0).to(device)
            fwt_weight = torch.empty(0).to(device)
            bwt_weight = torch.empty(0).to(device)
            for i in range(n):
                temp = torch.empty(0).to(device)
                for j in range(m):
                    temp = torch.cat([temp, bound_std * (self.weight[i][j] - self.shift[t-1][i][j]) / self.scale[t-1][i][j]], dim=0)

                weight = torch.cat([weight, temp], dim=1)
                fwt_weight = torch.cat([fwt_weight, self.weight[i][m]], dim=1)
            
            for j in range(m):
                bwt_weight = torch.cat([bwt_weight, self.weight[n][j]], dim=0)
            
            weight = F.dropout(weight, self.p, self.training)
            weight = torch.cat([torch.cat([weight, bwt_weight], dim=1), 
                                torch.cat([fwt_weight, self.weight[n][m]], dim=1)], dim=0)
            bias = self.bias[m] if self.bias is not None else None

        return weight, bias

    def get_optim_params(self):
        params = [self.weight[-1][-1]]
        for i in range(self.cur_task):
            params += [self.weight[i][-1], self.weight[-1][i]]
        if self.bias:
            params += [self.bias[-1]]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]
        return params

    def get_optim_scales(self, lr):
        params = []
        for i in range(self.cur_task):
            for j in range(self.cur_task):
                N = self.weight[i][j].numel()
                if N > 0:
                    N /= self.weight[i][j].shape[0]
                else:
                    N = 1
                params += [{'params':[self.scale[-1][i][j], self.shift[-1][i][j]], 'lr':lr/N}]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            for j in range(t+1):
                count += self.weight[i][j].numel()
        # for k in range(t+1):
        #     for i in range(k):
        #         for j in range(k):
        #             count += self.scale[k][i][j].numel()
        #     if self.bias:
        #         count += self.bias[k].numel()
        #     if self.norm_layer:
        #         if self.norm_layer.affine:
        #             count += self.norm_layer.weight[k].numel() + self.norm_layer.bias[k].numel()
        return count

    def norm_in(self):
        weight = torch.empty(0).to(device)
        for i in range(self.cur_task+1):
            weight = torch.cat([weight, self.weight[i][-1]], dim=1)
        norm = weight.norm(2, dim=self.dim_in)
        if self.bias is not None:
            norm = (norm ** 2 + self.bias[-1][self.shape_out[-2]:] ** 2) ** 0.5
        return norm

    def norm_out(self):   
        weight = torch.cat(self.weight[-1], dim=0)
        if self.s != 1:
            weight = weight.view(self.out_features, int(self.num_in[-1]/self.s/self.s), self.s, self.s)
        return weight.norm(2, dim=self.dim_out)

    def get_reg_strength(self):
        self.strength_in = self.weight[-1][-1].numel()
        self.strength_out = self.weight[-1][-1].numel()
        for i in range(self.cur_task):
            self.strength_in += self.weight[i][-1].numel()
            self.strength_out += self.weight[-1][i].numel()

        self.strength = (self.strength_in + self.strength_out)

    def squeeze(self, optim_state, mask_in=None, mask_out=None):

        if mask_out is not None:
            apply_mask_out(self.weight[-1][-1], mask_out, optim_state)
            for i in range(self.cur_task):
                apply_mask_out(self.weight[i][-1], mask_out, optim_state)

            self.num_out[-1] = self.weight[-1][-1].shape[0]
            self.out_features = sum(self.num_out)
            self.shape_out[-1] = self.out_features

            mask = torch.ones(self.shape_out[-2], dtype=bool, device=device)
            mask = torch.cat([mask, mask_out])

            if self.bias is not None:
                apply_mask_out(self.bias[-1], mask, optim_state)

            if self.norm_layer:
                if self.norm_layer.affine:
                    apply_mask_out(self.norm_layer.weight[-1], mask, optim_state)
                    apply_mask_out(self.norm_layer.bias[-1], mask, optim_state)

                if self.norm_layer.track_running_stats:
                    self.norm_layer.running_mean[-1] = self.norm_layer.running_mean[-1][mask]
                    self.norm_layer.running_var[-1] = self.norm_layer.running_var[-1][mask]

                self.norm_layer.num_features = self.out_features
                self.norm_layer.shape[-1] = self.out_features
        
        if mask_in is not None:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(self.weight[-1][-1], mask_in, optim_state)
            for i in range(self.cur_task):
                apply_mask_in(self.weight[-1][i], mask_in, optim_state)

            self.num_in[-1] = self.weight[-1][-1].shape[1]
            self.in_features = sum(self.num_in)
            self.shape_in[-1] = self.in_features
  
        self.get_reg_strength()

    def proximal_gradient_descent(self, lr, lamb, total_strength):
        eps = 0
        with torch.no_grad():
            strength_in = self.strength_in/total_strength
            strength_out = self.strength_out/total_strength
            strength = self.strength/total_strength
            # group lasso weights in
            if not self.last_layer:
                norm = self.norm_in()
                aux = 1 - lamb * lr * strength_in / norm
                aux = F.threshold(aux, 0, eps, False)
                self.mask_out = (aux > eps)
                self.weight[-1][-1].data *= aux.view(self.view_in)
                for i in range(self.cur_task):
                    self.weight[i][-1].data *= aux.view(self.view_in)
                if self.bias is not None:
                    self.bias[-1].data[self.shape_out[-2]:] *= aux

            # group lasso weights out
            if not self.first_layer:
                norm = self.norm_out()
                aux = 1 - lamb * lr * strength_out / norm
                aux = F.threshold(aux, 0, eps, False)
                self.mask_in = (aux > eps)
                if self.s != 1:
                    aux = aux.view(-1, 1, 1).expand(aux.size(0), self.s, self.s).contiguous().view(-1)
                self.weight[-1][-1].data *= aux.view(self.view_out)
                for i in range(self.cur_task):
                    self.weight[-1][i].data *= aux.view(self.view_out)    

            # group lasso affine weights
            if self.norm_layer:
                if self.norm_layer.affine:
                    norm = self.norm_layer.norm()
                    aux = 1 - lamb * lr * strength_in / norm
                    aux = F.threshold(aux, 0, eps, False)
                    self.mask_out *= (aux > eps)
                    self.norm_layer.weight[-1].data[self.norm_layer.shape[-2]:] *= aux
                    self.norm_layer.bias[-1].data[self.norm_layer.shape[-2]:] *= aux

class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(DynamicLinear, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

        self.view_in = [-1, 1]
        self.view_out = [1, -1]
        self.dim_in = [1]
        if s == 1:
            self.dim_out = [0]
        else:
            self.dim_out = [0, 2, 3]
            
        
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

        self.view_in = [-1, 1, 1, 1]
        self.view_out = [1, -1, 1, 1]
        self.dim_in = [1, 2, 3]
        self.dim_out = [0, 2, 3]

class DynamicClassifier(DynamicLinear):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(DynamicClassifier, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

    def forward(self, x, t, n, m):    
        weight, bias = self.get_parameters(t, n, m)
        x = F.linear(x, weight, bias)
        return x

    def expand(self, add_in=None, add_out=None, ablation='full'):
        self.cur_task += 1
        if add_in is None:
            add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features

        self.in_features += add_in
        self.out_features += add_out

        self.num_out.append(add_out)
        self.num_in.append(add_in)

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)

        self.weight.append([])
        if self.bias is not None:
            self.bias.append([])
        gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        for i in range(self.cur_task+1):
            bound_std = gain / math.sqrt(self.shape_in[i+1])
            self.weight[-1].append(nn.Parameter(torch.Tensor(self.num_out[-1], self.shape_in[i+1]).normal_(0, bound_std).to(device)))

            if self.bias is not None:
                self.bias[-1].append(nn.Parameter(torch.Tensor(self.num_out[-1]).uniform_(0, 0).to(device)))            
        # freeze old params
        if self.cur_task > 0:
            for i in range(self.cur_task):
                self.weight[-2][i].requires_grad = False
                if self.bias is not None:
                    self.bias[-2][i].requires_grad = False

        self.get_reg_strength()

    def get_parameters(self, t, n, m):
        weight = self.weight[m][n]
        bias = self.bias[m][n] if self.bias is not None else None
        return weight, bias

    def get_optim_params(self):
        params = []
        # params = [self.weight[-1][-1], self.bias[-1][-1]]-
        for i in range(self.cur_task+1):
            params += [self.weight[-1][i]]
            if self.bias:
                params += [self.bias[-1][i]]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            for j in range(i+1):
                count += self.weight[i][j].numel()
                if self.bias is not None:
                    count += self.bias[i][j].numel()
        return count

    def norm_out(self):   
        weight = self.weight[-1][-1][:, self.shape_in[-2]:]
        if self.s != 1:
            weight = weight.view(self.out_features, int(self.num_in[-1]/self.s/self.s), self.s, self.s)
        return weight.norm(2, dim=self.dim_out)

    def get_reg_strength(self):
        self.strength_out = self.weight[-1][-1].numel()
        for i in range(self.cur_task):
            self.strength_out += self.weight[-1][i].numel()

        self.strength = self.strength_out

    def squeeze(self, optim_state, mask_in=None, mask_out=None):
        if mask_in is not None:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            
            mask = torch.ones(self.shape_in[-2], dtype=bool, device=device)
            mask = torch.cat([mask, mask_in])
            apply_mask_in(self.weight[-1][-1], mask, optim_state)
            self.num_in[-1] = self.weight[-1][-1].shape[1]
            self.in_features = self.num_in[-1]
            self.shape_in[-1] = self.in_features
  
        self.get_reg_strength()

    def proximal_gradient_descent(self, lr, lamb, total_strength):
        eps = 0
        with torch.no_grad():
            strength_out = self.strength_out/total_strength
            # group lasso weights out
            norm = self.norm_out()
            aux = 1 - lamb * lr * strength_out / norm
            aux = F.threshold(aux, 0, eps, False)
            self.mask_in = (aux > eps)
            if self.s != 1:
                aux = aux.view(-1, 1, 1).expand(aux.size(0), self.s, self.s).contiguous().view(-1)
            self.weight[-1][-1].data[:, self.shape_in[-2]:] *= aux.view(self.view_out)   

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
            self.weight = []
            self.bias = []

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
            self.weight.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(1,1).to(device)))
            self.bias.append(nn.Parameter(torch.Tensor(self.num_features).uniform_(0,0).to(device)))
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

        if 'res' in self.norm_type:
            return input / (torch.sqrt(var.view(shape) + self.eps))
        else:
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

    def forward(self, input, t=-1, dropout=0.0):
        output = self.batch_norm(input, t)
        # if 'res' in self.norm_type:
        #     output = self.batch_norm(input, t) + input
        # else:
        #     output = self.batch_norm(input, t)

        if self.affine:
            weight = self.weight[t]
            bias = self.bias[t]
            if len(input.shape) == 4:
                output = output * weight.view(1,-1,1,1) + bias.view(1,-1,1,1)
            else:
                output = output * weight.view(1,-1) + bias.view(1,-1)

        return output


class re_sigma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, r):
        return sigma.clamp(min=r)

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
            