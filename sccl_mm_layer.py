
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

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
    
class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layer=None, bias=True, norm_type=None, smid=1, first_layer=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()

        self.base_in_features = in_features
        self.base_out_features = out_features
        self.first_layer = first_layer
        self.dropout = dropout
        if first_layer:
            self.in_features = in_features
        else:
            self.in_features = 0
        self.out_features = 0
        if bias:
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features))])
        else:
            self.register_parameter("bias", None)

        self.shape_in = [0]
        self.shape_out = [0]

        if norm_type:
            self.norm_layer = DynamicNorm(self.out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        # self.mask_pre_out = [None]
        # self.mask_pre_in = [None]
        # self.grad_in = 0
        # self.grad_out = 0

        self.old_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=False)
        if self.bias is not None:
            self.old_bias = nn.Parameter(torch.Tensor(self.out_features), requires_grad=False)
        else:
            self.old_bias = None

        self.next_layer = next_layer
        self.smid = smid
        self.mask = None
        

    def restrict_gradients(self, t, requires_grad):
        for i in range(0, t+1):
            self.weight[i].requires_grad = requires_grad
            self.fwt_weight[i].requires_grad = requires_grad
            self.bwt_weight[i].requires_grad = requires_grad
            if self.bias:
                self.bias[i].requires_grad = requires_grad
            if self.norm_layer is not None:
                if self.norm_layer.affine:
                    self.norm_layer.weight[i].requires_grad = requires_grad
                    self.norm_layer.bias[i].requires_grad = requires_grad

    def get_optim_params(self):
        params = [self.weight[-1], self.fwt_weight[-1], self.bwt_weight[-1]]
        if self.bias:
            params += [self.bias[-1]]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]
        return params

    def get_reg(self):
        reg = 0
        strength = self.strength_in + self.next_layer.strength_out
        reg += self.norm_in().sum() * strength
        reg += self.norm_out().sum() * strength
            
        if self.norm_layer:
            if self.norm_layer.affine:
                reg += self.norm_layer.reg().sum() * strength

        return reg, strength

    def get_importance(self):
        norm = self.norm_in() * self.norm_out()
        if self.norm_layer:
            if self.norm_layer.affine:
                norm *= self.norm_layer.reg()

        return norm

    def get_weights_in(self, t):
        return torch.cat([self.weight[t], self.fwt_weight[t]], dim=1)
    
    def get_weights_out(self, t):
        return torch.cat([self.next_layer.weight[t], self.next_layer.bwt_weight[t]], dim=0)

    def squeeze(self):
        if self.mask is not None:
            mask_out = self.mask
            self.weight[-1].data = self.weight[-1].data[mask_out].clone()
            self.weight[-1].grad = None
            self.fwt_weight[-1].data = self.fwt_weight[-1].data[mask_out].clone()
            self.fwt_weight[-1].grad = None 

            if self.bias:
                self.bias[-1].data = self.bias[-1].data[mask_out].clone()

            if self.norm_layer:
                self.norm_layer.squeeze(mask_out)

            self.out_features -= (mask_out.numel() - mask_out.sum()).item()
            self.shape_out[-1] = self.out_features

            if self.next_layer:
                if isinstance(self.next_layer, DynamicLinear) and isinstance(self, DynamicConv2D):
                    mask_in = self.mask.view(-1,1,1).expand(self.mask.size(0),self.next_layer.smid,self.next_layer.smid).contiguous().view(-1)
                else:
                    mask_in = self.mask

                self.next_layer.weight[-1].data = self.next_layer.weight[-1].data[:, mask_in].clone()
                self.next_layer.weight[-1].grad = None
                self.next_layer.bwt_weight[-1].data = self.next_layer.bwt_weight[-1].data[:, mask_in].clone()
                self.next_layer.bwt_weight[-1].grad = None
                self.next_layer.in_features -= (mask_in.numel() - mask_in.sum()).item()
                self.next_layer.shape_in[-1] = self.next_layer.in_features

    def expand(self, add_in=None, add_out=None):
        if add_in is None:
            add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features

        if isinstance(self, DynamicLinear):
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in)))
            # old neurons to new neurons
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features)))
            # new neurons to old neurons
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in)))
            fan_in = self.in_features + add_in
        else:
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size)))
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features // self.groups, *self.kernel_size)))
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in // self.groups, *self.kernel_size)))
            fan_in = (self.in_features + add_in) * np.prod(self.kernel_size)

        gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        bound = gain * math.sqrt(3.0/fan_in)
        nn.init.uniform_(self.weight[-1], -bound, bound)
        nn.init.uniform_(self.fwt_weight[-1], -bound, bound)
        nn.init.uniform_(self.bwt_weight[-1], -bound, bound)

        if self.bias:
            self.bias.append(nn.Parameter(torch.Tensor(add_out).uniform_(0, 0)))
        
        self.in_features += add_in
        self.out_features += add_out

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)

        if self.norm_layer:
            self.norm_layer.expand(add_out)

        self.strength_in = self.weight[-1].data.numel() + self.fwt_weight[-1].data.numel()
        self.strength_out = self.weight[-1].data.numel() + self.bwt_weight[-1].data.numel()
        
        self.mask = None
        # self.mask_pre_in.append(None)
        # self.mask_pre_out.append(None)

    def get_params(self, t):
        self.old_weight.data = self.weight[0].data
        if self.bias:
            self.old_bias.data = self.bias[0].data
        else:
            self.old_bias = None

        # for i in range(1, t+1):
        #   if self.mask_pre_in[i] is not None:
        #       if isinstance(self, DynamicLinear):
        #           fwt_weight = torch.zeros(self.weight[i].shape[0], self.old_weight.shape[1]).cuda()
        #           fwt_weight[:, self.mask_pre_in[i]] = self.fwt_weight[i].data
        #       else:
        #           fwt_weight = torch.zeros(self.weight[i].shape[0], self.old_weight.shape[1] // self.groups, *self.kernel_size).cuda()
        #           fwt_weight[:, self.mask_pre_in[i]] = self.fwt_weight[i].data
        #   else:
        #       fwt_weight = self.fwt_weight[i].data

        #   if self.mask_pre_out[i] is not None:
        #       if isinstance(self, DynamicLinear):
        #           bwt_weight = torch.zeros(self.old_weight.shape[0], self.weight[i].shape[1]).cuda()
        #           bwt_weight[self.mask_pre_out[i]] = self.bwt_weight[i].data
        #       else:
        #           bwt_weight = torch.zeros(self.old_weight.shape[0], self.weight[i].shape[1] // self.groups, *self.kernel_size).cuda()
        #           bwt_weight[self.mask_pre_out[i]] = self.bwt_weight[i].data
        #   else:
        #       bwt_weight = self.bwt_weight[i].data

        #   self.old_weight.data = torch.cat([torch.cat([self.old_weight.data, fwt_weight], dim=0), 
        #                                   torch.cat([bwt_weight, self.weight[i].data], dim=0)], dim=1)
        #   if self.bias is not None:
        #       self.old_bias.data = torch.cat([self.old_bias.data, self.bias[i].data])

        # if self.mask_pre_out[t+1] is not None:
        #   self.old_weight.data = self.old_weight.data[self.mask_pre_out[t+1]]
        #   if self.bias is not None:
        #       self.old_bias.data = self.old_bias.data[self.mask_pre_out[t+1]]

        # if self.mask_pre_in[t+1] is not None:
        #   self.old_weight.data = self.old_weight.data[:, self.mask_pre_in[t+1]]

        for i in range(1, t+1):
            fwt_weight = self.fwt_weight[i].data
            bwt_weight = self.bwt_weight[i].data

            self.old_weight.data = torch.cat([torch.cat([self.old_weight.data, fwt_weight], dim=0), 
                                            torch.cat([bwt_weight, self.weight[i].data], dim=0)], dim=1)
            if self.bias:
                self.old_bias.data = torch.cat([self.old_bias.data, self.bias[i].data])

        # if self.norm_layer:
        #     self.norm_layer.get_params(t)

    # def squeeze_previous(self, mask_in=None, mask_out=None):
    #     if mask_in is not None:
    #         self.fwt_weight[-1].data = self.fwt_weight[-1].data[:, self.mask_pre_in[-1]].clone()
    #         self.fwt_weight[-1].grad = None
    #         self.mask_pre_in[-1] = mask_in.clone()

    #     if mask_out is not None:
    #         self.bwt_weight[-1].data = self.bwt_weight[-1].data[self.mask_pre_out[-1]].clone()  
    #         self.bwt_weight[-1].grad = None
    #         self.mask_pre_out[-1] = mask_out.clone()


class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, next_layer=None, bias=True, norm_type=None, smid=1, first_layer=False, dropout=0.0):
        super(DynamicLinear, self).__init__(in_features, out_features, next_layer, bias, norm_type, smid, first_layer, dropout)
        
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, self.in_features))])
        self.fwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, 0))])
        self.bwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(0, self.in_features))])

        
    def forward(self, x, t, all=True):

        weight = torch.cat([torch.cat([self.old_weight, self.fwt_weight[t]], dim=0), torch.cat([self.bwt_weight[t], self.weight[t]], dim=0)], dim=1)
        if self.bias is not None:
            bias = torch.cat([self.old_bias, self.bias[t]])
        else:
            bias = None

        output = F.linear(x, weight, bias)

        if self.norm_layer is not None:
            output = self.norm_layer(output, t, self.dropout)

        if self.mask is not None:
            output[:, self.shape_out[-2]:] = output[:, self.shape_out[-2]:] * self.mask.view(1, -1)

        return output

    def norm_in(self, t=-1):
        weight = torch.cat([self.weight[t], self.fwt_weight[t]], dim=1)
        norm = weight.norm(2, dim=(1))
        if self.bias is not None:
            norm = (norm**2 + self.bias[t]**2)**(1/2)

        return norm

    def norm_out(self, t=-1):
        weight = torch.cat([self.next_layer.weight[t], self.next_layer.bwt_weight[t]], dim=0)
        norm = weight.norm(2, dim=0)
        return norm

    def get_grads_in(self, t):
        return torch.cat([self.weight[t].grad.data, self.fwt_weight[t].grad.data], dim=1)
    
    def get_grads_out(self, t):
        return torch.cat([self.next_layer.weight[t], self.next_layer.bwt_weight[t]], dim=0)

    def track_movement(self):
        return 
        
            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, next_layer, bias, norm_type, smid, first_layer, dropout):
        super(_DynamicConvNd, self).__init__(in_features, out_features, next_layer, bias, norm_type, smid, first_layer, dropout)
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
        
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, self.in_features // groups, *kernel_size))])
        self.fwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, 0 // groups, *kernel_size))])
        self.bwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(0, self.in_features // groups, *kernel_size))])


class DynamicConv2D(_DynamicConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, next_layer=None, bias=True, norm_type=None, smid=1, first_layer=False, dropout=0.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, next_layer, bias, norm_type, smid, first_layer, dropout)
    
    def forward(self, x, t):

        weight = torch.cat([torch.cat([self.old_weight, self.fwt_weight[t]], dim=0), torch.cat([self.bwt_weight[t], self.weight[t]], dim=0)], dim=1)
        if self.bias is not None:
            bias = torch.cat([self.old_bias, self.bias[t]])
        else:
            bias = None

        output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        if self.norm_layer is not None:
            output = self.norm_layer(output, t, self.dropout)

        if self.mask is not None:
            output[:, self.shape_out[-2]:] = output[:, self.shape_out[-2]:] * self.mask.view(1, -1, 1, 1)

        return output


    def norm_in(self, t=-1):
        weight = self.get_weights_in(t)
        norm = weight.norm(2, dim=(1,2,3))
        if self.bias is not None:
            norm = (norm**2 + self.bias[t]**2)**(1/2)

        return norm

    def norm_out(self, t=-1):
        weight = self.get_weights_out(t)
        if isinstance(self.next_layer, DynamicLinear):
            weight = weight.view(self.next_layer.weight[t].shape[0] + self.next_layer.bwt_weight[t].shape[0], 
                                self.weight[t].shape[0], 
                                self.next_layer.smid, self.next_layer.smid)

        norm = weight.norm(2, dim=(0,2,3))
        return norm

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

        if self.affine:
            self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(0).uniform_(1,1))])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(0).uniform_(0,0))])
            self.old_weight = nn.Parameter(torch.Tensor(0), requires_grad=False)
            self.old_bias = nn.Parameter(torch.Tensor(0), requires_grad=False)

        if self.track_running_stats:
            self.running_mean = [torch.zeros(0).to(device)]
            self.running_var = [torch.ones(0).to(device)]
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

        if self.track_running_stats:
            self.running_mean.append(torch.zeros(self.num_features).to(device))
            self.running_var.append(torch.ones(self.num_features).to(device))
            self.num_batches_tracked = 0

        self.strength = 2*add_num
            

    def squeeze(self, mask=None):
        if mask is not None:
            mask_temp = torch.ones(self.shape[-2]).bool().to(device)
            mask = torch.cat([mask_temp, mask])
            if self.affine:
                self.weight[-1].data = self.weight[-1].data[mask].clone()
                self.bias[-1].data = self.bias[-1].data[mask].clone()

            self.num_features -= (mask.numel() - mask.sum()).item()
            self.shape[-1] = self.num_features

            if self.track_running_stats:
                self.running_mean[-1] = self.running_mean[-1][mask]
                self.running_var[-1] = self.running_var[-1][mask]
    
    def reg(self):
        return (self.weight[-1][self.shape[-2]:]**2 + self.bias[-1][self.shape[-2]:]**2) ** 1/2

    def get_params(self, t):
        if not self.affine:
            return

        self.old_weight.data = self.weight[0].data
        self.old_bias.data = self.bias[0].data

        for i in range(1, t+1):
            self.old_weight.data = torch.cat([self.old_weight.data, self.weight[i].data])
            self.old_bias.data = torch.cat([self.old_bias.data, self.bias[i].data])

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
        if self.norm_type == 'bn':
            output = self.batch_norm(input, t)
        elif self.norm_type =='gn':
            output = self.layer_norm(input)
        elif self.norm_type =='bn_gn':
            output = self.layer_norm(self.batch_norm(input, t) + input)

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
    
            