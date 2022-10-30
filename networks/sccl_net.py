from tkinter.tix import Tree
from turtle import forward
from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
from layers.sccl_mm_layer import DynamicLinear, DynamicConv2D, _DynamicLayer

from utils import *
import sys

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DynamicModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.permute = [None]

    def restrict_gradients(self, t, requires_grad):
        for m in self.DM:
            m.restrict_gradients(t, requires_grad)

    def get_optim_params(self):
        params = []
        for m in self.DM:
            params += m.get_optim_params()
        return params

    def get_optim_scales(self, lr):
        params = []
        for m in self.DM:
            params += m.get_optim_scales(lr)
        return params

    def expand(self, new_class, ablation='full'):
        for m in self.DM[:-1]:
            m.expand(add_in=None, add_out=None, ablation=ablation)
        self.DM[-1].expand(add_in=None, add_out=new_class, ablation=ablation)

    def squeeze(self, optim_state):
        self.total_strength = 1
        for m in self.DM[:-1]:
            m.squeeze(optim_state)
            # self.total_strength += m.strength

    def forward(self, input, t=-1, assemble=False):
        if t == -1:
            t = len(self.DM[-1].shape_out)-1

        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                input = module(input, t, assemble)
            else:
                input = module(input)

        return input

    def count_params(self, t=-1):
        if t == -1:
            t = len(self.DM[-1].shape_out)-1
        model_count = 0
        layers_count = []
        print('num neurons:', end=' ')
        for m in self.DM:
            print(m.out_features, end=' ')
            count = m.count_params(t)
            model_count += count
            layers_count.append(count)

        print('|num params:', model_count, end=' | ')

        return model_count, layers_count

    def group_lasso_reg(self):
        total_reg = 0
        for i, m in enumerate(self.DM[:-1]):
            reg = m.get_reg()
            total_reg += reg
                            
        return total_reg/self.total_strength

    def proximal_gradient_descent(self, lr, lamb):
        for m in self.DM[:-1]:
            m.proximal_gradient_descent(lr, lamb, self.total_strength)

    def report(self):
        for m in self.DM:
            print(m.__class__.__name__, m.in_features, m.out_features)
            
class MLP(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 400
        self.layers = nn.ModuleList([
            nn.Flatten(),
            # nn.Dropout(0.25),
            DynamicLinear(np.prod(input_size), N, first_layer=True, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicLinear(N, N, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicLinear(N, 0, bias=True, last_layer=True),
            ])
        
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]

class VGG8(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, bias=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size

        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, first_layer=True, bias=bias),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            ])

        s = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D):
                s = compute_conv_output_size(s, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
            elif isinstance(m, nn.MaxPool2d):
                s = compute_conv_output_size(s, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(128*s*s, 256, norm_type=norm_type, s=s),
            nn.ReLU(),
            DynamicLinear(256, 0, last_layer=True)
            ])

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]

class VGG(_DynamicModel):
    '''
    VGG model 
    '''
    def __init__(self, input_size, cfg, norm_type=None):
        super(VGG, self).__init__()

        nchannels, size, _ = input_size

        self.layers = make_layers(cfg, nchannels, norm_type=norm_type)

        s = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D):
                s = compute_conv_output_size(s, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
            elif isinstance(m, nn.MaxPool2d):
                s = compute_conv_output_size(s, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(512*s*s, 4096, s=s),
            nn.ReLU(True),
            DynamicLinear(4096, 4096),
            nn.ReLU(True),
            DynamicLinear(4096, 0, last_layer=True),
        ])

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]


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
            DynamicConv2D(ncha,64,kernel_size=size//8, first_layer=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool2d(2),

            DynamicConv2D(64,128,kernel_size=size//10, norm_type=norm_type),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            DynamicConv2D(128,256,kernel_size=2, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(2),
            ])

        s = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D):
                s = compute_conv_output_size(s, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
            elif isinstance(m, nn.MaxPool2d):
                s = compute_conv_output_size(s, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(256*s*s, 2048, s=s, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.5),
            DynamicLinear(2048, 2048, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.5),
            DynamicLinear(2048, 0, last_layer=True, norm_type=norm_type)
        ])
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]

'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return DynamicConv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return DynamicConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(_DynamicModel):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.ModuleList([
            DynamicConv2D(in_planes, planes, kernel_size=3, 
                                stride=stride, padding=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            DynamicConv2D(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, norm_type=norm_type)
        ])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        else:
            self.shortcut = None

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]


    def forward(self, x, t):
        out = x.clone()
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                out = module(out, t)
                if out is None:
                    out = 0
                    break
            else:
                out = module(out)

        if self.shortcut:
            out += self.shortcut(x, t)
        else:
            out += x

        return F.relu(out)

class Bottleneck(_DynamicModel):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type=None):
        super(Bottleneck, self).__init__()

        self.layers = nn.ModuleList([
            DynamicConv2D(in_planes, planes, kernel_size=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            DynamicConv2D(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            DynamicConv2D(planes, self.expansion * planes, 
                                kernel_size=1, bias=False, norm_type=norm_type)
        ])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        else:
            self.shortcut = None

        self.DM = [m for m in self.layers if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]


    def forward(self, x, t):
        out = x.clone()
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                out = module(out, t)
                if out is None:
                    out = 0
                    break
            else:
                out = module(out)

        if self.shortcut:
            out += self.shortcut(x, t)
        else:
            out += x

        return F.relu(out)

class ResNet(_DynamicModel):
    def __init__(self, block, num_blocks, norm_type, input_size, nf=32):
        super(ResNet, self).__init__()
        n_channels, in_size, _ = input_size
        self.in_planes = nf

        self.conv1 = DynamicConv2D(n_channels, nf*1, kernel_size=3,
                               stride=1, padding=1, bias=False, norm_type=norm_type, first_layer=True)
        self.blocks = self._make_layer(block, nf*1, num_blocks[0], stride=1, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*2, num_blocks[1], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*4, num_blocks[2], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*8, num_blocks[3], stride=2, norm_type=norm_type)
        self.linear = DynamicLinear(nf*8*block.expansion, 0, last_layer=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

        m = self.conv1
        for block in self.blocks:
            m.next_layers = [block.layers[0]]
            if block.shortcut:
                m.next_layers.append(block.shortcut)
            m = block.layers[-1]
        m.next_layers = [self.linear]


    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride, norm_type))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(blocks)

    def forward(self, x, t):
        out = F.relu(self.conv1(x, t))
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out, t)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out, t)
        return out

    def squeeze(self, optim_state):
        self.total_strength = 1
        # for m in self.DM[:-1]:
        #     self.total_strength += m.strength
        # share masks between skip connection layers
        if self.conv1.mask is None:
            return
        share_mask = self.conv1.mask

        for i, block in enumerate(self.blocks):
            if block.shortcut:
                share_mask = block.shortcut.mask
                            
            share_mask += block.layers[-1].mask
            block.layers[-1].mask = share_mask

        for m in self.DM[:-1]:
            m.squeeze(optim_state)
            # self.total_strength += m.strength

def ResNet18(input_size, norm_type=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type, input_size)

def ResNet34(input_size, norm_type=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type, input_size)

def ResNet50(input_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_type, input_size)

def ResNet101(input_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], norm_type, input_size)

def ResNet152(input_size, norm_type=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], norm_type, input_size)

