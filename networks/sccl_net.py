import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer, _DynamicModel

from utils import *
import sys

            
class MLP(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 400
        self.layers = nn.ModuleList([
            nn.Flatten(),
            # nn.Dropout(0.25),
            DynamicLinear(np.prod(input_size), N, first_layer=True, dropout=0.25, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicLinear(N, N, dropout=0.25, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicLinear(N, 0, bias=True),
            ])
        
        self.DM = [m for m in self.layers if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layer = self.DM[i+1]


class VGG8(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, bias=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size

        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, first_layer=True, bias=bias),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, dropout=0.25, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, dropout=0.25, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, dropout=0.5, bias=bias),
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
            DynamicLinear(128*self.smid*self.smid, 256, smid=self.smid, norm_type=norm_type, dropout=0.0),
            nn.ReLU(),
            DynamicLinear(256, 0)
            ])

        self.DM = [m for m in self.layers if isinstance(m, _DynamicLayer)]
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
            DynamicLinear(4096, 4096),
            nn.ReLU(True),
            DynamicLinear(4096, 0),
        ])

        self.DM = [m for m in self.layers if isinstance(m, _DynamicLayer)]
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
            DynamicLinear(2048, 2048),
            nn.ReLU(),
            DynamicLinear(2048, 0)
        ])
        self.DM = [m for m in self.layers if isinstance(m, _DynamicLayer)]
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


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 norm_layer(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(_DynamicModel):
#     def __init__(self, input_size, block, num_blocks):
#         super(ResNet, self).__init__()

#         nchannels, size, _ = input_size
#         self.in_planes = 64
#         self.conv1 = nn.Conv2d(nchannels, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)

#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self._norm_layer))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def ResNet18(**kwargs: Any):
#     return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def ResNet34(**kwargs: Any):
#     return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def ResNet50(**kwargs: Any):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def ResNet101(**kwargs: Any):
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# def ResNet152(**kwargs: Any):
#     return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


