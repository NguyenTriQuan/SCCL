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
import cvxpy as cp
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
        self.next_ks = 1
        self.res = False
        self.gain = torch.nn.init.calculate_gain('relu')
        

    def forward(self, x): 
        if isinstance(self, DynamicConv2D):
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.linear(x, self.weight, self.bias)

        if self.norm_layer is not None:
            output = self.norm_layer(output)

        return output

    def initialize(self):  
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weight)
        # fan = self.out_features * self.next_ks
        self.bound = self.gain / math.sqrt(fan_in)
        nn.init.normal_(self.weight, 0, self.bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def normalize(self):
        with torch.no_grad():
            mean = self.weight.mean(dim=self.norm_dim).detach().view(self.norm_view)
            var = self.weight.var(dim=self.norm_dim, unbiased=False).detach().sum() * self.next_ks
            std = var ** 0.5
            self.weight.data = self.gain * (self.weight.data) / std 
        
    def optimize_layerwise(self, inp, alpha=0.7, 
                    beta=0.001, max_param_per_kernel=None, 
                    min_param_to_node=None,
                    init_weight=None,
                    node_constraint=False):
        # print('Optimizing layerwise sparse pattern')
        is_conv = False

        # Params in layer 
        n_params = int(math.ceil((1-self.sparsity)*self.weight.numel())) # This has to be integer
        
        # The value of input nodes is descirbed by P_in
        if isinstance(self, DynamicConv2D):
            C_out, C_in, kernel_size, kernel_size = self.weight.shape
            min_param_per_kernel = int(math.ceil(n_params/(C_in*C_out))) 
            if max_param_per_kernel is None:
                max_param_per_kernel = kernel_size*kernel_size
            # Ensure enough params to assign to valid the sparsity
            elif max_param_per_kernel < min_param_per_kernel:
                max_param_per_kernel = min_param_per_kernel
            else:   # it's oke
                pass
            
            if min_param_to_node is None:
                min_param_to_node = 1
            # Ensure the valid of eff node constraint
            elif min_param_to_node > min_param_per_kernel:    
                min_param_to_node = min_param_per_kernel
            else:   # it's oke
                pass
            
            P_in = torch.sum(inp, dim=(1,2)).numpy()
            is_conv = True
        else:
            C_out, C_in = self.weight.shape
            kernel_size = 1
            max_param_per_kernel = kernel_size
            min_param_to_node = 1
            # P_in = torch.sum(inp, dim=)
            if len(inp.shape) != 1:
                P_in = torch.sum(inp, dim=1).numpy()
            else:
                P_in = inp.numpy()
            if len(P_in.shape) != 1 and P_in.shape[0] != C_out:
                raise ValueError('Wrong input dimension')
        
        # Mask variable 
        M = cp.Variable((C_in, C_out), integer=True)

        scaled_M = None
        if init_weight is not None:
            if is_conv:
                mag_orders = init_weight.transpose(1,0).view(C_in, C_out, -1).abs().argsort(dim=-1, descending=True).numpy()
                init_weight = torch.sum(init_weight, dim=(2,3)).transpose(1,0).numpy()
            else:
                init_weight = init_weight.transpose(1,0).numpy()
            init_weight = np.abs(init_weight)
            # scaled_M = cp.multiply(M, init_weight)

        # Sun 
        sum_in = cp.sum(M, axis=1)
        sum_out = cp.sum(M, axis=0)
        # sum_in = cp.sum(M, axis=1) * P_in
        # sum_out = cp.sum(cp.diag(P_in)@M, axis=0)


        # If eff_node_in is small which means there is a large number of input effective node 
        inv_eff_node_in = cp.sum(cp.pos(min_param_to_node - sum_in))
        inv_eff_node_out = cp.sum(cp.pos(min_param_to_node - sum_out))

        # OPtimize nodes 
        max_nodes = C_in + C_out
        A = max_nodes - (inv_eff_node_in + inv_eff_node_out) 
        # A = A / max_nodes   # Scale to 1

        # Optimize paths
        # B = (cp.sum(P_in @ M)) / cp.sum(P_in)   # Downscale with input nodes' values
        min_out_node = int(n_params/(C_out * max_param_per_kernel))
        remainder = n_params - min_out_node * (C_out * max_param_per_kernel)
        try:
            max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel)) + \
                        remainder * np.sort(P_in)[-(min_out_node+1)]
        except:
            max_path = np.sum(np.sort(P_in)[-min_out_node:] * (C_out * max_param_per_kernel))
        
        if scaled_M is not None:
            B = (cp.sum(P_in @ scaled_M)) 
            # B = (cp.sum(P_in @ scaled_M)) / np.sum(P_in)
        else:
            B = (cp.sum(P_in @ M)) / max_path
            A = A / max_nodes
        # C = (cp.sum(P_in @ M)) / max_path
        # Regulaziration
        Reg = (n_params-cp.sum(cp.pos(1-M))) / n_params     # maximize number of edges 
        # Reg = 0


        # Constraint the total activated params 
        if self.sparsity == 0:
            constraint = [cp.sum(M) <= n_params, M <= max_param_per_kernel, M >= 0] 
        else:
            constraint = [cp.sum(M) <= n_params, M <= max_param_per_kernel, M >= 0, 
                # cp.sum(M, axis=0) >= int(C_out*(1-sparsity)), #int(C_out*0.1)
                # cp.sum(M, axis=1) >= int(C_in*(1-sparsity))] #int(C_in*0.1)
                # cp.max(cp.sum(M, axis=0)) <= int(C_in*max_param_per_kernel**2*(1-sparsity)), # max params to a output node < C_in
                # cp.max(cp.sum(M, axis=1)) <= int(C_out*max_param_per_kernel**2*(1-sparsity)), # max params from a input node < C_out
                # cp.max(cp.sum(M, axis=0)) <= C_in,
                # cp.max(cp.sum(M, axis=1)) <= C_out,
                ]
        if node_constraint:
            constraint.append(
                cp.max(cp.sum(M, axis=0)) <= int(C_in*max_param_per_kernel**2*(1-self.sparsity))
            )
            constraint.append(
                cp.max(cp.sum(M, axis=1)) <= int(C_out*max_param_per_kernel**2*(1-self.sparsity))
            )
        # Objective function
        # alpha = 0.7
        obj =cp.Maximize(alpha * A + (1-alpha) * B + beta * Reg)

        # Init problem
        prob = cp.Problem(obj, constraint)

        # Solving
        prob.solve()
        # prob.value

        if is_conv:
            a = torch.tensor(M.value, dtype=torch.int16)
            mat = []
            for i in range(C_out):
                row = []
                for j in range(C_in):
                    try:
                        r = np.zeros(kernel_size**2)
                        if init_weight is not None:
                            one_idxs = mag_orders[j,i][:a[j,i]]
                            r[one_idxs] = 1 
                        else:
                            r[:a[j,i]] = 1
                            np.random.shuffle(r)
                        row.append(r.reshape(kernel_size, kernel_size))
                    except:
                        print(r)
                        print(a[j,i])
                mat.append(row)
            mat = np.array(mat)
            self.weight.data.copy_(torch.tensor(mat))
        else:
            self.weight.data.copy_(torch.tensor(M.value).transpose(1,0))

        actual_sparsity = 1 - self.weight.sum().item() / self.weight.numel()
        # print(f'Desired sparsity is {sparsity} and optimizer finds sparsity is {actual_sparsity}')

        

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


    
            