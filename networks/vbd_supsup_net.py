'''
Modified from https://github.com/pytorch/vision.git
'''
import math
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Bernoulli, LogNormal, Normal
import torch.nn.functional as F
from utils import *


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VBD_Layer(nn.Module):
    """docstring for ClassName"""
    def __init__(self, in_features):
        super(VBD_Layer, self).__init__()
        self.in_features = in_features
        self.mu = nn.Parameter(torch.Tensor(in_features).uniform_(1.0, 1.0))
        self.log_sigma2 = nn.Parameter(torch.Tensor(in_features).uniform_(-5.0, -5.0))
        self.normal = Normal(0, 1)
        self.thres = 3.0

    def forward(self, x): 

        if len(x.shape) == 2:
            shape = (1, -1)
        else:
            shape = (1, -1, 1, 1)
        if self.training:
            sigma = torch.exp(0.5*self.log_sigma2)
            epsilon = self.normal.sample(self.log_sigma2.size()).cuda()
            x = x * (self.mu + sigma * epsilon).view(shape)
        else:
            x = x * self.mu.view(shape)
        
        return x

    def kl_divergence(self):
        kld = 0.5*torch.log1p(self.mu*self.mu/(torch.exp(self.log_sigma2)+1e-8))
        return kld.sum()

    @property
    def log_alpha(self):
        return self.log_sigma2 - 2.0 * torch.log(self.mu + 1e-8)

    @property
    def p(self):
        alpha = torch.exp(self.log_alpha)
        return alpha/(1+alpha)

    def get_mask(self):
        return (self.log_alpha < self.thres)




class MLP(nn.Module):

    def __init__(self, input_size, taskcla, mul=1):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 1000
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(np.prod(input_size), N, bias=False),
            VBD_Layer(N),
            nn.ReLU(),
            nn.Linear(N, N, bias=False),
            VBD_Layer(N),
            nn.ReLU(),
            ])
        self.last = nn.ModuleList([nn.Linear(N, ncla) for t, ncla in taskcla])

        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.requires_grad = False
                # m.bias.requires_grad = False

    def forward(self, x, t):
        for m in self.layers:
            x = m(x)

        return self.last[t](x)




class VGG8(nn.Module):

    def __init__(self, input_size, taskcla, mul=1, norm_layer=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size

        self.layers = nn.ModuleList([
            nn.Conv2d(nchannels, 32, kernel_size=3, padding=1, bias=True),
            VBD_Layer(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            VBD_Layer(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            VBD_Layer(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            VBD_Layer(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            VBD_Layer(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            VBD_Layer(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(128*self.smid*self.smid, 256, bias=True),
            VBD_Layer(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            ])

        self.last = nn.ModuleList([nn.Linear(256, ncla) for t, ncla in taskcla])

        # for m in self.layers:
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False

    def forward(self, x, t):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return self.last[t](x)


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, input_size, cfg, norm_layer=False):
        super(VGG, self).__init__()

        nchannels, size, _ = input_size

        self.layers = make_layers(cfg, nchannels, norm_layer=norm_layer)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(512*self.smid*self.smid, 4096, smid=self.smid),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 0),
        ])



def make_layers(cfg, nchannels, norm_layer=False):
    layers = []
    in_channels = nchannels
    layers += nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True)
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def VGG11(input_size):
    """VGG 11-layer model (configuration "A")"""
    return VGG(input_size, cfg['A'], norm_layer=False)


def VGG11_BN(input_size):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(input_size, cfg['A'], norm_layer=True)


def VGG13(input_size):
    """VGG 13-layer model (configuration "B")"""
    return VGG(input_size, cfg['B'], norm_layer=False)


def VGG13_BN(input_size):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(input_size, cfg['B'], batch_norm=True)


def VGG16(input_size):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size, cfg['C'], norm_layer=False)


def VGG16_BN(input_size):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(input_size, cfg['C'], norm_layer=True)


def VGG19(input_size):
    """VGG 19-layer model (configuration "E")"""
    return VGG(input_size, cfg['D'], norm_layer=False)


def VGG19_BN(input_size):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(input_size, cfg['D'], norm_layer=True)


class Alexnet(nn.Module ):

    def __init__(self, input_size, mul=1):
        super(Alexnet,self).__init__()

        ncha, size, _ = input_size
        self.mul = mul

        self.layers = nn.ModuleList([
            nn.Conv2d(ncha,64,kernel_size=size//8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64,128,kernel_size=size//10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(128,256,kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            ])

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(256*self.smid*self.smid, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 0)
        ])
