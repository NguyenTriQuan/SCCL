import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
from layers.dae_layer import DynamicLinear, DynamicConv2D, _DynamicLayer, DynamicClassifier

from utils import *
import sys
from arguments import get_args
args = get_args()
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _DynamicModel(nn.Module):
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.permute = []

    def get_optim_params(self):
        params = []
        for m in self.DM:
            params += m.get_optim_params()
        return params

    def get_optim_scales(self, lr):
        params = []
        for m in self.DM[:-1]:
            params += m.get_optim_scales(lr)
        return params

    def expand(self, new_class, ablation='full'):
        self.total_strength = 1
        self.DM[0].expand(add_in=0, add_out=None, ablation=ablation)
        for m in self.DM[1:-1]:
            m.expand(add_in=None, add_out=None, ablation=ablation)
        self.DM[-1].expand(add_in=None, add_out=new_class, ablation=ablation)
        for i, m in enumerate(self.DM[:-1]):
            self.DM[i].strength = self.DM[i].strength_in + self.DM[i+1].strength_out
            self.total_strength += self.DM[i].strength

    def squeeze(self, optim_state):
        self.total_strength = 1
        mask_in = None
        mask_out = self.DM[0].mask_out
        self.DM[0].squeeze(optim_state, mask_in, mask_out)
        mask_in = mask_out
        i = 1
        for m in self.DM[1:-1]:
            mask_out = self.DM[i].mask_out
            m.squeeze(optim_state, mask_in, mask_out)
            mask_in = mask_out
            i += 1
        self.DM[-1].squeeze(optim_state, mask_in, None)
        for i, m in enumerate(self.DM[:-1]):
            self.DM[i].strength = self.DM[i].strength_in + self.DM[i+1].strength_out
            self.total_strength += self.DM[i].strength

    def forward(self, input, t, mask=False, mem=False):
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                input = module(input, t, mask, mem)                    
            else:
                input = module(input)
        return input

    def count_params(self, t=-1):
        if t == -1:
            t = len(self.DM[-1].shape_out)-2
        model_count = 0
        layers_count = []
        print('| num neurons:', end=' ')
        for m in self.DM:
            print(m.out_features, end=' ')
            count = m.count_params(t)
            model_count += count
            layers_count.append(count)

        print('| num params:', model_count, end=' |')
        # print(layers_count)
        print()
        return model_count, layers_count

    def proximal_gradient_descent(self, lr, lamb):
        for m in self.DM[:-1]:
            m.proximal_gradient_descent(lr, lamb, self.total_strength)

    def freeze(self, t):
        for m in self.DM[:-1]:
            m.freeze(t)
    
    def update_scale(self):
        for m in self.DM[:-1]:
            m.update_scale()
    
    def get_old_params(self, t):
        # get old parameters for task t
        for m in self.DM[:-1]:
            m.get_old_params(t)
        
    def get_mem_params(self):
        for m in self.DM:
            m.get_mem_params()

    def report(self):
        for m in self.DM:
            print(m.__class__.__name__, m.in_features, m.out_features)
            
class MLP(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 200
        p = 1
        if 'drop_arch' in args.ablation:
            p = 0
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Dropout(0.25*p),
            DynamicLinear(np.prod(input_size), N, first_layer=True, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicLinear(N, N, bias=True, norm_type=norm_type),
            nn.ReLU(),
            # DynamicLinear(N, N, bias=True, norm_type=norm_type),
            # nn.ReLU(),
            # DynamicLinear(N, N, bias=True, norm_type=norm_type),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            DynamicClassifier(N, 0, bias=True),
            ])
        
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

class VGG8(_DynamicModel):

    def __init__(self, input_size, mul=1, norm_type=None, bias=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size
        p = 1
        if 'drop_arch' in args.ablation:
            p = 0
        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, first_layer=True, bias=bias),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.25*p),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.25*p),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5*p),
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
            # nn.Dropout(0.5),
            DynamicClassifier(256, 0, last_layer=True)
            ])

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
            DynamicClassifier(int(4096*mul), 0, last_layer=True),
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
        p = 1
        if 'drop_arch' in args.ablation:
            p = 0
        self.layers = nn.ModuleList([
            DynamicConv2D(ncha,64,kernel_size=size//8, first_layer=True, norm_type=norm_type),
            nn.ReLU(),
            nn.Dropout(0.2*p),
            nn.MaxPool2d(2),

            DynamicConv2D(64,128,kernel_size=size//10, norm_type=norm_type),
            nn.Dropout(0.2*p),
            nn.ReLU(),
            nn.MaxPool2d(2),

            DynamicConv2D(128,256,kernel_size=2, norm_type=norm_type),
            nn.ReLU(),
            nn.Dropout(0.5*p),
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
            nn.Dropout(0.5*p),
            DynamicLinear(2048, 2048, norm_type=norm_type),
            nn.ReLU(),
            nn.Dropout(0.5*p),
            DynamicLinear(2048, 0, last_layer=True, norm_type=norm_type)
        ])
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]
        for i, m in enumerate(self.DM[:-1]):
            m.next_layers = [self.DM[i+1]]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, norm_type=None):
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
        activation=args.activation,
        norm_type=norm_type
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, norm_type=None):
    """1x1 convolution"""
    return DynamicConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, activation=args.activation, norm_type=norm_type)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_type = None,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, norm_type=norm_type)
        self.conv2 = conv3x3(planes, planes, norm_type=norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if args.res:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_type = None,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, norm_type=norm_type)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, norm_type=norm_type)
        self.conv3 = conv1x1(width, planes * self.expansion, norm_type=norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if args.res:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        norm_type,
        input_size,
        output_size,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
    ) -> None:
        super().__init__()
        
        self._norm_layer = norm_type

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = DynamicConv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, norm_type=norm_type, activation='identity')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DynamicLinear(512 * block.expansion, output_size, bias=True, activation=args.activation)
        self.WN = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

        for i, m in enumerate(self.WN[:-1]):
            self.WN[i].next_ks = self.WN[i+1].ks
            print(self.WN[i].next_ks)
        
        self.initialize()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_type = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if args.res:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, norm_type=norm_type),
                )            

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_type
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_type=norm_type,
                )
            )

        return nn.Sequential(*layers)

    def normalize(self):
        for m in self.WN:
            m.normalize()
    
    def initialize(self):
        print('initialize')
        for m in self.WN:
            m.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet18(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type, input_size, output_size)

def ResNet34(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type, input_size, output_size)

def ResNet50(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_type, input_size, output_size)

def ResNet101(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], norm_type, input_size, output_size)

def ResNet152(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], norm_type, input_size, output_size)

