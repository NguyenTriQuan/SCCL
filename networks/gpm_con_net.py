# from tkinter.tix import Tree
# from turtle import forward
# from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
# from layers.sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
from layers.gpm_con_layer import DynamicLinear, DynamicConv2D, _DynamicLayer

from utils import *
import sys
from arguments import get_args
args = get_args()
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DynamicModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.features_mean = None
        self.features_mean_mask = None
        self.features_mean_mem = None

        self.features_var = None
        self.features_var_mask = None
        self.features_var_mem = None
        self.ncla = [0]

    def forward(self, input):
        for module in self.layers:
            input = module(input)
        return input

    def count_GPM(self):
        gpm_count = 0
        for m in self.DM:
            if m.projection_matrix is not None:
                gpm_count += m.feature.numel()
        print('GPM count:', gpm_count)

    def project_gradient(self):
        for m in self.DM:
            m.project_gradient()

    def get_feature(self, thresholds):
        for i, m in enumerate(self.DM):
            m.get_feature(thresholds[i])

    def track_input(self, tracking):
        for m in self.DM:
            m.act = None
            m.track_input = tracking

    def normalize(self):
        for m in self.DM:
            m.normalize()
            
class MLP(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, ncla=0):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 200
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Dropout(0.25),
            DynamicLinear(np.prod(input_size), N, bias=False, norm_type=norm_type),
            nn.ReLU(),
            DynamicLinear(N, N, bias=False, norm_type=norm_type),
            nn.ReLU(),
            DynamicLinear(N, args.feat_dim, bias=False),
            ])
        self.classifier = nn.Linear(args.feat_dim, ncla, bias=True)
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

class VGG8(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, ncla=0):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size
        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=False),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=False, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=False),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=False, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=False),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=False, dropout=0.5),
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
            nn.Dropout(0.5),
            DynamicLinear(256, args.feat_dim, bias=False)
            ])
        self.classifier = nn.Linear(args.feat_dim, ncla, bias=True)
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

class VGG(_DynamicModel):
    '''
    VGG model 
    '''
    def __init__(self, input_size, cfg, norm_type=None, mul=1):
        super(VGG, self).__init__()

        nchannels, size, _ = input_size

        self.layers = make_layers(cfg, nchannels, norm_type=norm_type, mul=mul)

        self.p = 0.1
        s = size
        for m in self.layers:
            if isinstance(m, DynamicConv2D):
                s = compute_conv_output_size(s, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
            elif isinstance(m, nn.MaxPool2d):
                s = compute_conv_output_size(s, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(int(512*s*s*mul), int(4096*mul), s=s),
            # nn.Dropout(self.p),
            nn.ReLU(True),
            DynamicLinear(int(4096*mul), int(4096*mul)),
            # nn.Dropout(self.p),
            nn.ReLU(True),
            DynamicLinear(int(4096*mul), 0, last_layer=True),
        ])

        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]


def make_layers(cfg, nchannels, norm_type=None, bias=True, mul=1):
    layers = []
    in_channels = nchannels
    layers += DynamicConv2D(in_channels, int(cfg[0]*mul), kernel_size=3, padding=1, norm_type=norm_type, bias=bias, first_layer=True), nn.ReLU(inplace=True)
    in_channels = int(cfg[0]*mul)
    p = 0.1
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v*mul)
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

def VGG16_small(input_size, norm_type):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size, cfg['C'], norm_type=norm_type, mul=0.5)

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
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            DynamicConv2D(64,128,kernel_size=size//10, norm_type=norm_type),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),

            DynamicConv2D(128,256,kernel_size=2, norm_type=norm_type),
            nn.Dropout(0.5),
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
            nn.Dropout(0.5),
            DynamicLinear(2048, 2048, norm_type=norm_type),
            nn.Dropout(0.5),
            DynamicLinear(2048, args.feat_dim, last_layer=True, activation='identity', norm_type=None)
        ])
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            self.DM[i].next_ks = self.DM[i+1].ks
            print(self.DM[i].next_ks)

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

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
        #                     kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        # else:
        #     self.shortcut = None
        self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False, norm_type=norm_type)

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

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
        #                   kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        # else:
        #     self.shortcut = None
        self.shortcut = DynamicConv2D(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False, norm_type=norm_type)

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
        s_mid = 1
        # if in_size == 84:
        #     s_mid = 2

        self.in_planes = nf

        self.conv1 = DynamicConv2D(n_channels, nf*1, kernel_size=3,
                               stride=1, padding=1, bias=False, norm_type=norm_type, first_layer=True)
        self.blocks = self._make_layer(block, nf*1, num_blocks[0], stride=1, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*2, num_blocks[1], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*4, num_blocks[2], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*8, num_blocks[3], stride=2, norm_type=norm_type)
        self.linear = DynamicLinear(nf*8*block.expansion*s_mid*s_mid, 0, last_layer=True, s=s_mid)
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
        # out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out, t)
        return out

    def squeeze(self, optim_state):
        if self.conv1.mask is None:
            return

        for i, block in enumerate(self.blocks):
            share_mask = block.shortcut.mask + block.layers[-1].mask
            block.shortcut.mask = share_mask
            block.layers[-1].mask = share_mask
        self.total_strength = 1
        for m in self.DM[:-1]:
            m.squeeze(optim_state)
            self.total_strength += m.strength

    # def expand(self, new_class, ablation='full'):
    #     for m in self.DM[:-1]:
    #         m.expand(add_in=None, add_out=None, ablation=ablation)
    #     self.DM[-1].expand(add_in=None, add_out=new_class, ablation=ablation)

    #     self.total_strength = 1
    #     for m in self.DM[:-1]:
    #         m.get_reg_strength()
    #         self.total_strength += m.strength

    #     share_strength_in = self.conv1.strength_in
    #     share_strength_out = self.conv1.strength_out
    #     share_strength = self.conv1.strength
    #     share_layers = []
    #     for i, block in enumerate(self.blocks):
    #         if block.shortcut:
    #             for layer in share_layers:
    #                 layer.strength_in = share_strength_in
    #                 layer.strength_out = share_strength_out
    #                 layer.strength = share_strength
    #             share_strength = block.shortcut.strength
    #             share_strength_in = block.shortcut.strength_in
    #             share_strength_out = block.shortcut.strength_out
    #             share_layers = []
                            
    #         share_layers.append(block.layers[-1])
    #         share_strength = max(block.layers[-1].strength, share_strength)
    #         share_strength_in = max(block.layers[-1].strength_in, share_strength_in)
    #         share_strength_out = max(block.layers[-1].strength_out, share_strength_out)
    #     for layer in share_layers:
    #         layer.strength_in = share_strength_in
    #         layer.strength_out = share_strength_out
    #         layer.strength = share_strength
    #     share_layers = []

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

