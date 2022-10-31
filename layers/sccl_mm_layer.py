# from pyexpat import features
# from re import S
# from tkinter.tix import InputOnly
# from traceback import print_tb
# from sqlalchemy import false
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
# from arguments import get_args
# args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.dropout = 0.0
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.in_features = 0
        self.out_features = 0
            
        self.weight = [None]
        self.fwt_weight = [None]
        self.bwt_weight = [None]
        self.aux_weight = [None]
        if bias:
            self.bias = [None]
            self.aux_bias = [None]
        else:
            self.register_parameter("bias", None)

        self.w_sigma = [None]
        self.bwt_sigma = [None]
        self.fwt_sigma = [None]

        self.shape_in = [self.in_features]
        self.shape_out = [self.out_features]

        self.norm_type = norm_type

        if norm_type:
            self.norm_layer = DynamicNorm(self.out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        self.next_layers = next_layers # where output of this layer go
        self.s = s
        self.mask = None
        
        self.cur_task = 0

    def forward(self, x, t, assemble=False):    
        weight, bias = self.get_parameters(t)

        if weight.numel() == 0:
            return None

        if isinstance(self, DynamicLinear):
            output = F.linear(x, weight, bias)
            # if not self.last_layer and assemble:
            #     _, n_features = output.shape
            #     output = output.view(t, -1, n_features)
            #     for i in range(t):
            #         output[i, :, self.shape_out[i+1]:] = 0
            #     output = output.view(-1, n_features)
        else:
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            # if not self.last_layer and assemble:
            #     _, n_channels, s, s = output.shape
            #     output = output.view(t, -1, n_channels, s, s)
            #     for i in range(t):
            #         output[i, :, self.shape_out[i+1]:] = 0

            #     output = output.view(-1, n_channels, s, s)

        if self.norm_layer is not None:
            output = self.norm_layer(output, t, self.dropout)
        return output

    def get_parameters(self, t):
        weight = torch.empty(0).to(device)
        for i in range(1, t):
            bwt_weight = torch.cat([torch.empty(0).to(device)] + [self.bwt_weight[i][j] * self.bwt_sigma[t][i][j]
                                    for j in range(1, i)], dim=0)
            fwt_weight = torch.cat([torch.empty(0).to(device)] + [self.fwt_weight[i][j] * self.fwt_sigma[t][i][j]
                                    for j in range(1, i)], dim=1)
                
            weight = torch.cat([torch.cat([weight, bwt_weight], dim=1), 
                                torch.cat([fwt_weight, self.weight[i] * self.w_sigma[t][i]], dim=1)], dim=0)

        bwt_weight = torch.cat([torch.empty(0).to(device)] + [self.bwt_weight[t][j] for j in range(1, t)], dim=0)
        fwt_weight = torch.cat([torch.empty(0).to(device)] + [self.fwt_weight[t][j] for j in range(1, t)], dim=1)
        weight = torch.cat([torch.cat([weight, bwt_weight], dim=1), 
                            torch.cat([fwt_weight, self.weight[t]], dim=1)], dim=0)

        if self.bias:
            bias = self.bias[t]
        else:
            bias = None

        if self.last_layer and t < self.cur_task:
            weight = torch.cat([weight] + [self.aux_weight[i][t] for i in range(t+1, self.cur_task+1)], dim=0)
            if self.bias:
                bias = torch.cat([bias] + [self.aux_bias[i][t] for i in range(t+1, self.cur_task+1)], dim=0)

        return weight, bias

    def get_optim_params(self):
        params = []
        params += [self.weight[-1]] + self.fwt_weight[-1][1:] + self.bwt_weight[-1][1:]
        if self.last_layer:
            params += self.aux_weight[-1][1:]
        if self.bias:
            params += [self.bias[-1]]
            if self.last_layer:
                params += self.aux_bias[-1][1:]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]

        return params

    def get_optim_scales(self, lr):
        params = []
        for i in range(1, self.cur_task):
            N = self.weight[i].numel()
            N = max(N, 1)
            params += [{'params':[self.w_sigma[-1][i]], 'lr':lr/N}]
            for j in range(1, i):
                N = self.bwt_weight[i][j].numel()
                N = max(N, 1)
                params += [{'params':[self.bwt_sigma[-1][i][j]], 'lr':lr/N}]

                N = self.fwt_weight[i][j].numel()
                N = max(N, 1)
                params += [{'params':[self.fwt_sigma[-1][i][j]], 'lr':lr/N}]

        return params

    def count_params(self, t):
        count = 0
        for k in range(1, t+1):
            count += self.weight[k].numel()
            for i in range(1, k):
                count += self.fwt_weight[k][i].numel() + self.bwt_weight[k][i].numel()
                count += self.w_sigma[k][i].numel()
                if self.last_layer:
                    count += self.aux_weight[k][i].numel()
                    if self.bias:
                        count += self.aux_bias[k][i].numel()
                for j in range(1, i):
                    count += self.fwt_sigma[k][i][j].numel() + self.bwt_sigma[k][i][j].numel()
            if self.bias:
                count += self.bias[k].numel()
            if self.norm_layer:
                if self.norm_layer.affine:
                    count += self.norm_layer.weight[k].numel() + self.norm_layer.bias[k].numel()
        return count

    def norm_in(self):
        fwt_weight = torch.cat([torch.empty(0).to(device)] + [self.fwt_weight[-1][j] for j in range(1, self.cur_task)], dim=1)
        weight = torch.cat([fwt_weight, self.weight[-1]], dim=1)
        norm = weight.norm(2, dim=self.dim_in)
        if self.bias is not None:
            norm = (norm ** 2 + self.bias[-1][self.shape_out[-2]:] ** 2) ** 0.5
        return norm

    def norm_out(self, i):
        bwt_weight = torch.cat([torch.empty(0).to(device)] + [self.next_layers[i].bwt_weight[-1][j] for j in range(1, self.cur_task)], dim=0)
        weight = torch.cat([bwt_weight, self.next_layers[i].weight[-1]], dim=0)
        if isinstance(self, DynamicConv2D) and isinstance(self.next_layers[i], DynamicLinear):
            weight = weight.view(self.next_layers[i].out_features, 
                                self.weight[-1].shape[i], self.next_layers[i].s, self.next_layers[i].s)
        return weight.norm(2, dim=self.dim_out)

    def get_reg_strength(self):
        self.strength_in = self.weight[-1].data.numel()
        for i in range(1, self.cur_task):
            self.strength_in += self.fwt_weight[-1][i].data.numel()
        self.strength_out = [0]
        for m in self.next_layers:
            strength_out = m.weight[-1].data.numel()
            for i in range(1, m.cur_task):
                strength_out += m.bwt_weight[-1][i].data.numel()
            self.strength_out.append(strength_out)
        self.strength_out = max(self.strength_out)

        self.strength = (self.strength_in + self.strength_out)

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
            mask_out = self.mask
            apply_mask_out(self.weight[-1], mask_out)
            for i in range(1, self.cur_task):
                apply_mask_out(self.fwt_weight[-1][i], mask_out)

            self.out_features = self.shape_out[-2] + self.weight[-1].shape[0]
            self.shape_out[-1] = self.out_features

            mask = torch.ones(self.shape_out[-2]).bool().to(device)
            mask = torch.cat([mask, mask_out])

            if self.bias:
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
                if isinstance(m, DynamicLinear) and isinstance(self, DynamicConv2D):
                    mask_in = self.mask.view(-1,1,1).expand(self.mask.size(0),m.s,m.s).contiguous().view(-1)
                else:
                    mask_in = self.mask

                apply_mask_in(m.weight[-1], mask_in)
                for i in range(1, m.cur_task):
                    apply_mask_in(m.bwt_weight[-1][i], mask_in)

                m.in_features = m.shape_in[-2] + m.weight[-1].shape[1]
                m.shape_in[-1] = m.in_features
  
            self.mask = None
            self.get_reg_strength()

    def expand(self, add_in=None, add_out=None, ablation='full'):
        self.cur_task += 1
        if add_in is None:
            add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features

        if self.first_layer:
            if self.cur_task == 1:
                add_in = self.base_in_features
            else:
                add_in = 0

        self.fwt_weight.append([None])
        self.bwt_weight.append([None])
        if self.last_layer:
            self.aux_weight.append([None])
            self.aux_bias.append([None])
        if isinstance(self, DynamicConv2D):
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).to(device)))
            for i in range(1, self.cur_task):
                # connect neurons of old task i to neurons of new task
                self.fwt_weight[-1].append(nn.Parameter(torch.Tensor(add_out, (self.shape_in[i]-self.shape_in[i-1]) // self.groups, *self.kernel_size).to(device)))
                # connect neurons of new task to neurons of old task i
                self.bwt_weight[-1].append(nn.Parameter(torch.Tensor(self.shape_out[i]-self.shape_out[i-1], add_in // self.groups, *self.kernel_size).to(device)))
                # head for old model i to predict for new task
                if self.last_layer:
                    self.aux_weight[-1].append(nn.Parameter(torch.Tensor(add_out, self.shape_in[i] // self.groups, *self.kernel_size).to(device)))
                    if self.bias:
                        self.aux_bias[-1].append(nn.Parameter(torch.Tensor(add_out).uniform_(0, 0).to(device)))
            K = np.prod(self.kernel_size)
        else:
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).to(device)))
            for i in range(1, self.cur_task):
                # connect neurons of old task i to neurons of new task
                self.fwt_weight[-1].append(nn.Parameter(torch.Tensor(add_out, self.shape_in[i]-self.shape_in[i-1]).to(device)))
                # connect neurons of new task to neurons of old task i
                self.bwt_weight[-1].append(nn.Parameter(torch.Tensor(self.shape_out[i]-self.shape_out[i-1], add_in).to(device)))
                # head for old model i to predict for new task
                if self.last_layer:
                    self.aux_weight[-1].append(nn.Parameter(torch.Tensor(add_out, self.shape_in[i]).to(device)))
                    if self.bias:
                        self.aux_bias[-1].append(nn.Parameter(torch.Tensor(add_out).uniform_(0, 0).to(device)))
            K = 1

        fan_in = (self.in_features + add_in) * K

        if fan_in != 0:
            # init
            gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            bound_std = gain / math.sqrt(fan_in)
            nn.init.normal_(self.weight[-1], 0, bound_std)
            for i in range(1, self.cur_task):
                nn.init.normal_(self.fwt_weight[-1][i], 0, bound_std)
                nn.init.normal_(self.bwt_weight[-1][i], 0, bound_std)
                if self.last_layer:
                    nn.init.normal_(self.aux_weight[-1][i], 0, gain / math.sqrt(self.shape_in[i] * K))

            # rescale old tasks params
            if 'scale' not in ablation and not self.last_layer:
                self.w_sigma.append([None])
                self.bwt_sigma.append([None])
                self.fwt_sigma.append([None])
                for i in range(1, self.cur_task):
                    self.bwt_sigma[-1].append([None])
                    self.fwt_sigma[-1].append([None])
                    if self.weight[i].numel() == 0:
                        self.w_sigma[-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
                    else:
                        w_std = self.weight[i].std()
                        self.w_sigma[-1].append(nn.Parameter(bound_std/w_std))

                    for j in range(1, i):
                        if self.bwt_weight[i][j].numel() == 0:
                            self.bwt_sigma[-1][-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
                        else:
                            bwt_std = self.bwt_weight[i][j].std()
                            self.bwt_sigma[-1][-1].append(nn.Parameter(bound_std/bwt_std))

                        if self.fwt_weight[i][j].numel() == 0:
                            self.fwt_sigma[-1][-1].append(nn.Parameter(torch.ones(1).to(device), requires_grad=False))
                        else:
                            fwt_std = self.fwt_weight[i][j].std()
                            self.fwt_sigma[-1][-1].append(nn.Parameter(bound_std/fwt_std))
            else:
                self.w_sigma.append([nn.Parameter(torch.ones(1).to(device), requires_grad=False) for i in range(self.cur_task)])
                self.bwt_sigma.append([[nn.Parameter(torch.ones(1).to(device), requires_grad=False) for j in range(i)] 
                                                                                                    for i in range(self.cur_task)])
                self.fwt_sigma.append([[nn.Parameter(torch.ones(1).to(device), requires_grad=False) for j in range(i)]
                                                                                                    for i in range(self.cur_task)])
        self.in_features += add_in
        self.out_features += add_out

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)

        if self.bias:
            self.bias.append(nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0).to(device)))

        if self.norm_layer:
            self.norm_layer.expand(add_out)             

        # freeze old params
        if self.cur_task > 1:
            self.weight[-2].requires_grad = False
            for i in range(1, self.cur_task-1):
                self.fwt_weight[-2][i].requires_grad = False
                self.bwt_weight[-2][i].requires_grad = False
                self.w_sigma[-2][i].requires_grad = False
                for j in range(1, i):
                    self.fwt_sigma[-2][i][j].requires_grad = False
                    self.bwt_sigma[-2][i][j].requires_grad = False   
            if self.bias:
                self.bias[-2].requires_grad = False
            if self.norm_layer:
                if self.norm_layer.affine:
                    self.norm_layer.weight[-2].requires_grad = False
                    self.norm_layer.bias[-2].requires_grad = False

    def proximal_gradient_descent(self, lr, lamb, total_strength):

        with torch.no_grad():
            strength_in = self.strength_in/total_strength
            strength_out = self.strength_out/total_strength
            strength = self.strength/total_strength
            # group lasso weights in
            norm = self.norm_in()
            aux = 1 - lamb * lr * strength_in / norm
            aux = F.threshold(aux, 0, 0, False)
            self.mask = (aux > 0)

            self.weight[-1].data *= aux.view(self.view_in)
            for i in range(1, self.cur_task):
                self.fwt_weight[-1][i].data *= aux.view(self.view_in)
            if self.bias:
                self.bias[-1].data[self.shape_out[-2]:] *= aux

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
                    for i in range(1, self.cur_task):
                        m.bwt_weight[-1][i].data *= aux.view(m.view_out)                  
                self.mask *= mask_temp
            # group lasso affine weights
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
            self.weight = [None]
            self.bias = [None]

        if self.track_running_stats:
            self.running_mean = [None]
            self.running_var = [None]
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
            var = input.var([0, 2, 3], unbiased=False)
            shape = (1, -1, 1, 1)
        else:
            mean = input.mean([0])
            var = input.var([0], unbiased=False)
            shape = (1, -1)

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

        # output = self.batch_norm(input, t) + self.layer_norm(input)
        if 'res' in self.norm_type:
            output = self.batch_norm(input, t) + input
        else:
            output = self.batch_norm(input, t)

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
    
            