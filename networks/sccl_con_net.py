import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
from sccl_con_layer import DynamicLinear, DynamicConv2D, _DynamicLayer

from utils import *
import sys

class _DynamicModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.permute = [None]
        self.ncla = [0]
        self.repres_mean = None
        self.repres_std = None

    def restrict_gradients(self, t, requires_grad):
        for m in self.DM:
            m.restrict_gradients(t, requires_grad)

    def get_optim_params(self):
        params = []
        for m in self.DM:
            params += m.get_optim_params()
        return params
    def get_params(self, t):
        for m in self.DM:
            m.get_params(t)

    def expand(self, new_class):
        self.DM[0].expand(add_in=None, add_out=None)
        for m in self.DM[1:-1]:
            m.expand(add_in=None, add_out=None)
        self.DM[-1].expand(add_in=None, add_out=None)
        self.ncla.append(self.ncla[-1] + new_class)

    def group_lasso_reg(self):
        total_reg = 0
        total_strength = 0
        for i, m in enumerate(self.DM[:-1]):

            reg, strength = m.get_reg()
            total_reg += reg
            total_strength += strength
                            
        return total_reg/total_strength

    def forward(self, input, t=-1):
        features = input
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                features = module(features, t)
            else:
                features = module(features)

        return features

    def compute_model_size(self, t=-1):
        model_count = 0
        layers_count = []
        for m in self.DM:
            temp_count = 0
            for p in m.weight[:t]:
                temp_count += p.numel()
            temp_count += m.weight[t].numel()

            for p in m.fwt_weight[:t]:
                temp_count += p.numel()
            temp_count += m.fwt_weight[t].numel()

            for p in m.bwt_weight[:t]:
                temp_count += p.numel()
            temp_count += m.bwt_weight[t].numel()

            if m.bias is not None:
                for p in m.bias[:t]:
                    temp_count += p.numel()
                temp_count += m.bias[t].numel()

            if m.norm_layer is not None:
                if m.norm_layer.affine:
                    for p in m.norm_layer.weight[:t]:
                        temp_count += p.numel()
                    temp_count += m.norm_layer.weight[t].numel()

                    for p in m.norm_layer.bias[:t]:
                        temp_count += p.numel()
                    temp_count += m.norm_layer.bias[t].numel()

            model_count += temp_count
            layers_count.append(temp_count)

        return model_count, layers_count

    def track_gradient(self, sbatch):
        for i, m in enumerate(self.DM[:-1]):

            grad_in = m.sum_grad_in()
            if isinstance(m, DynamicConv2D) and isinstance(self.DM[i+1], DynamicLinear):
                grad_out = self.DM[i+1].sum_grad_out(size=(self.DM[i+1].old_weight.shape[0] + self.DM[i+1].fwt_weight[-1].shape[0], 
                                                    m.old_weight.shape[0], 
                                                    self.smid, self.smid))  
            else:
                grad_out = self.DM[i+1].sum_grad_out()

            m.grad_in -= grad_in*sbatch
            m.grad_out -= grad_out*sbatch

    def s_H(self, t=-1):
        s_H = 1
        for m in self.DM:
            weight = torch.cat([torch.cat([m.old_weight, m.fwt_weight[t]], dim=0), torch.cat([m.bwt_weight[t], m.weight[t]], dim=0)], dim=1)
            # s_H *= weight.norm(p='fro')
            # s_H *= weight.abs().max()
            s_H *= weight.norm(p=2).detach().item()
        return s_H

class MLP(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 400
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Dropout(0.25),
            DynamicLinear(np.prod(input_size), N, bias=True, norm_type=norm_type, first_layer=True),
            nn.ReLU(),
            DynamicLinear(N, N, bias=True, last_layer=True),
            ])
        
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layer = self.DM[i+1]


class VGG8(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, bias=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size

        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, first_layer=False),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(128*self.smid*self.smid, 256, smid=self.smid, last_layer=False),
            ])

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layer = self.DM[i+1]

class VGG(_DynamicModel):
    '''
    VGG model 
    '''
    def __init__(self, input_size, cfg, norm_type=None):
        super(VGG, self).__init__()

        nchannels, size, _ = input_size

        self.layers = make_layers(cfg, nchannels, norm_type=norm_type)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(512*self.smid*self.smid, 4096, smid=self.smid),
            nn.ReLU(True),
            DynamicLinear(4096, 4096, last_layer=True),
        ])

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layer = self.DM[i+1]


def make_layers(cfg, nchannels, norm_type=None, bias=True):
    layers = []
    in_channels = nchannels
    layers += DynamicConv2D(in_channels, cfg[0], kernel_size=3, padding=1, norm_type=norm_type, bias=bias, first_layer=True), nn.ReLU(inplace=True)
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [DynamicConv2D(in_channels, v, kernel_size=3, padding=1, norm_type=norm_type, bias=bias), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def VGG11(input_size, norm_type=None):
    """VGG 11-layer model (configuration "A")"""
    return VGG(input_size, cfg['A'], norm_type=norm_type)

def VGG13(input_size, norm_type):
    """VGG 13-layer model (configuration "B")"""
    return VGG(input_size, cfg['B'], norm_type=norm_type)

def VGG16(input_size, norm_type):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size, cfg['C'], norm_type=norm_type)

def VGG19(input_size, norm_type):
    """VGG 19-layer model (configuration "E")"""
    return VGG(input_size, cfg['D'], norm_type=norm_type)


class Alexnet(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None):
        super(Alexnet,self).__init__()

        ncha, size, _ = input_size
        self.mul = mul

        self.layers = nn.ModuleList([
            DynamicConv2D(ncha,64,kernel_size=size//8, first_layer=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            DynamicConv2D(64,128,kernel_size=size//10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            DynamicConv2D(128,256,kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(256*self.smid*self.smid, 2048),
            nn.ReLU(),
            DynamicLinear(2048, 2048, last_layer=True),
        ])
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layer = self.DM[i+1]

'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''


class BasicBlock(_DynamicModel):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.layers = nn.ModuleList([
            DynamicConv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, batch_norm=True),
            nn.ReLU(),
            DynamicConv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, batch_norm=True),
            nn.ReLU(),
            ])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = DynamicConv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, batch_norm=True)


    def forward(self, x, t=-1):
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                x = module(x, t)
            else:
                x = module(x)

        if isinstance(self.shortcut, _DynamicLayer):
            out += self.shortcut(x, t)
        else:
            out += x
        out = F.relu(out)
        return out
    def forward_mask(self, x, masks):
        pass
