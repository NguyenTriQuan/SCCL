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

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0, activation='leaky_relu'):
        super(_DynamicLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.in_features = in_features
        self.out_features = out_features

        bias = False
        self.bias = nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0)) if bias else None
        self.track_input = False
        self.projection_matrix = None
        self.feature = None 
        self.next_ks = 1
        
        if activation == 'leaky_relu':
            self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope)
            self.activation = nn.LeakyReLU(args.negative_slope, inplace=False)
        else:
            self.gain = 1
            self.activation = nn.Identity()

    def forward(self, x):  
        if self.track_input:
            self.act = x.detach()  
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, self.weight, self.bias)

        if self.norm_layer is not None:
            output = self.norm_layer(output)

        output = self.activation(output)
        return output

    def get_feature(self, threshold):
        mat = self.get_mat()
        if self.feature is None:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            # criteria (Eq-5)
            sval_ratio = S**2
            sval_ratio = sval_ratio/sval_ratio.sum()
            r = (torch.cumsum(sval_ratio, dim=0) < threshold).sum().item() #+1 
            self.feature = U[:, :r]
        else:
            U1, S1, Vh1 = torch.linalg.svd(mat, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = mat - torch.mm(torch.mm(self.feature, self.feature.T), mat)
            U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            # update GPM
            self.feature = torch.cat([self.feature, U[:, 0: r]], dim=1)
            self.feature = self.feature[:, 0: self.feature.shape[0]]

        self.projection_matrix = torch.mm(self.feature, self.feature.T)

    def project_gradient(self):
        if self.projection_matrix is None:
            return
        sz =  self.weight.size(0)
        projected_grad = torch.mm(self.weight.grad.data.view(sz,-1), self.projection_matrix).view(self.weight.size())
        self.weight.grad.data -= projected_grad

    def initialize(self):  
        # fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weight)
        fan = self.out_features * self.next_ks
        self.bound = self.gain / math.sqrt(fan)
        nn.init.normal_(self.weight, 0, self.bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def normalize(self):
        with torch.no_grad():
            mean = self.weight.mean(dim=self.norm_dim).detach().view(self.norm_view)
            var = self.weight.var(dim=self.norm_dim, unbiased=False).detach().sum() * self.next_ks
            std = var ** 0.5
            self.weight.data = self.gain * (self.weight.data-mean) / std 
        

class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0, activation='leaky_relu'):
        super(DynamicLinear, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout, activation)

        self.norm_dim = (1)
        self.norm_view = (-1, 1)
        self.ks = 1
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.initialize()

        if norm_type is not None:
            self.norm_layer = nn.BatchNorm1d(out_features, track_running_stats=False, affine=False)
        else:
            self.norm_layer = None

    def get_mat(self):
        return self.act.T
            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout, activation):
        super(_DynamicConvNd, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout, activation)
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
                stride=1, padding=0, dilation=1, groups=1, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0, activation='leaky_relu'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout, activation)

        self.norm_dim = (1, 2, 3)
        self.norm_view = (-1, 1, 1, 1)
        self.ks = np.prod(self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features // self.groups, *self.kernel_size))
        self.initialize()

        if norm_type is not None:
            self.norm_layer = nn.BatchNorm2d(out_features, track_running_stats=False, affine=False)
        else:
            self.norm_layer = None
    
    def get_mat(self):
        k = 0
        batch_size = self.act.shape[0]
        s = compute_conv_output_size(self.act.shape[-1], self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0])
        mat = torch.zeros((self.kernel_size[0]*self.kernel_size[1]*self.in_features, s*s*batch_size)).to(device)
        for kk in range(batch_size):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=self.act[kk,:,ii:self.kernel_size[0]+ii,jj:self.kernel_size[1]+jj].reshape(-1) 
                    k +=1
        return mat

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

        if self.affine:
            weight = self.weight[t]
            bias = self.bias[t]
            if len(input.shape) == 4:
                output = output * weight.view(1,-1,1,1) + bias.view(1,-1,1,1)
            else:
                output = output * weight.view(1,-1) + bias.view(1,-1)

        return output


    
            