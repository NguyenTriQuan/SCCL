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
from layers.npb_cl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer

from utils import *
import sys
from arguments import get_args
args = get_args()
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_intermediate_inputs(net, input_):
    
    def get_children(model: torch.nn.Module):
        # get children form model!
        children = list(model.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return model
        else:
            # look for children from children... to the last child!
            for child in children:
                    if isinstance(child, nn.BatchNorm2d) or \
                        isinstance(child, nn.ReLU) or \
                        isinstance(child, nn.AdaptiveAvgPool2d):
                        continue
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children

    flatt_children = get_children(net)

    visualization = []

    def hook_fn(m, i, o):
        visualization.append(i)

    for layer in flatt_children:
        layer.register_forward_hook(hook_fn)
        
    # for param in net.parameters():
    #     param.data.copy_(torch.ones_like(param))

    out = net(input_)  
    
    return visualization

class _DynamicModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(_DynamicModel, self).__init__()
        self.ncla = [0]

    def forward(self, input, t):
        for module in self.layers:
            if isinstance(module, _DynamicLayer):
                input = module(input, t)
            else:
                input = module(input)
        input = self.last[t](input)
        return input

    def normalize(self):
        for m in self.DM:
            m.normalize()

    def ERK_sparsify(self, sparsity=0.9):
        print(f'initialize by ERK, sparsity {sparsity}')
        density = 1 - sparsity
        erk_power_scale = 1

        total_params = 0
        for m in self.DM:
            total_params += m.weight.numel()
        is_epsilon_valid = False

        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            for i, m in enumerate(self.DM):
                m.raw_probability = 0
                n_param = np.prod(m.weight.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if m in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    m.raw_probability = (np.sum(m.weight.shape) / np.prod(m.weight.shape)) ** erk_power_scale
                    divisor += m.raw_probability * n_param
            epsilon = rhs / divisor
            max_prob = np.max([m.raw_probability for m in self.DM])
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for m in self.DM:
                    if m.raw_probability == max_prob:
                        # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(m)
            else:
                is_epsilon_valid = True

        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for i, m in enumerate(self.DM):
            n_param = np.prod(m.weight.shape)
            if m in dense_layers:
                m.sparsity = 0
            else:
                probability_one = epsilon * m.raw_probability
                m.sparsity = 1 - probability_one
            print(
                f"layer: {i}, shape: {m.weight.shape}, sparsity: {m.sparsity}"
            )
            total_nonzero += (1-m.sparsity) * m.weight.numel()
        print(f"Overall sparsity {1-total_nonzero / total_params}")
        #     mask.data.copy_((torch.rand(mask.shape) < density_dict[name]).float().data.cuda())

        #     total_nonzero += density_dict[name] * mask.numel()
        # print(f"Overall sparsity {total_nonzero / total_params}")

    def prune(self, input_shape, sparsity, alpha, beta, node_constraint=False, 
                max_param_per_kernel=None, min_param_to_node=None, is_store_mask=False, file_name=None):
        layer_id = 0

        c, h, w = input_shape

        input_ = torch.ones((2,c,h,w)).double()
        self.cpu()
        self.double()
        print(input_.shape)
        prev = input_

        self.ERK_sparsify(sparsity)
        for i, m in enumerate(self.DM):

            output = torch.ones((2,c,h,w)).double()
            n = 0
            for layer in self.layers:
                if isinstance(layer, _DynamicLayer):
                    if n == i:
                        prev = output.detach().clone()
                        break
                    output = layer(output)
                    n += 1
                else:
                    output = layer(output)
            
            print(prev.shape, prev.sum().item()/prev.numel())

            if isinstance(m, DynamicConv2D):
                print(f'Considering layer {i}')
                if i == 0: # Input layer
                    m.optimize_layerwise(prev[0], alpha=alpha, beta=beta, 
                                        max_param_per_kernel=None)
                else:
                    if m.res:
                        m.optimize_layerwise(prev[0], alpha=self.alpha, 
                                    node_constraint=node_constraint)
                    else:
                        m.optimize_layerwise(prev[0], alpha=alpha, beta=beta, 
                                    max_param_per_kernel=max_param_per_kernel,
                                    min_param_to_node=min_param_to_node, 
                                    node_constraint=node_constraint)
                # actual_sparsity = 1 - mask.sum().item() / mask.numel()
                # print(f'Desired sparsity is {sparsity_dict[name]} and optimizer finds sparsity is {actual_sparsity}')

                # if 'shortcut' in name:
                #     if self.max_param_per_kernel > 5:
                #         self.max_param_per_kernel -= 2
                        
            else:    # Linear layer
                print(f'Considering layer {i}')
                m.optimize_layerwise(prev[0], alpha=alpha, beta=0)

            print(m.weight.sum().item())
        

        # cloned_net = fine_tune_mask(cloned_net, input_shape)
        # count_ineff_param(cloned_net, input_shape)

        # cloned_net.float()

        # # Copy mask
        # cloned_net.to(self.device)
        # for (n, m), (name, mask) in zip(net.named_buffers(), cloned_net.named_buffers()):
        #     m.copy_(mask)

        # if is_store_mask:
        #     if file_name is not None:
        #         try:
        #             store_mask(net, file_name)
        #         except:
        #             raise RuntimeError('There is something wrong with store mask function!')
        #     else:
        #         print('No store mask file name')
            
class MLP(_DynamicModel):

    def __init__(self, input_size, taskcla, mul=1, norm_type=None, ncla=0, bias=True):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 200
        self.layers = nn.ModuleList([
            nn.Flatten(),
            DynamicLinear(np.prod(input_size), N, bias=bias, norm_type=norm_type),
            nn.ReLU(),
            DynamicLinear(N, N, bias=bias, norm_type=norm_type),
            nn.ReLU(),
            ])
        self.last = []
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(N,n))
        self.DM = [m for m in self.modules() if isinstance(m, _DynamicLayer)]

class VGG8(_DynamicModel):

    def __init__(self, input_size, taskcla, mul=1, norm_type=None, ncla=0, bias=True):
        super(VGG8, self).__init__()

        nchannels, size, _ = input_size
        self.mul = mul
        self.input_size = input_size
        self.layers = nn.ModuleList([
            DynamicConv2D(nchannels, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(32, 32, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.25),

            DynamicConv2D(32, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(64, 64, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.25),

            DynamicConv2D(64, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(128, 128, kernel_size=3, padding=1, norm_type=norm_type, bias=bias, dropout=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

            DynamicConv2D(128, 256, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
            DynamicConv2D(256, 256, kernel_size=3, padding=1, norm_type=norm_type, bias=bias),
            nn.ReLU(),
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
            DynamicLinear(256*s*s, 256, norm_type=norm_type, s=s),
            nn.ReLU(),
            # nn.Dropout(0.5),
            ])
        self.last = []
        for t,n in taskcla:
            self.last.append(torch.nn.Linear(256,n))
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
            nn.ReLU(True),
            DynamicLinear(int(4096*mul), int(4096*mul)),
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

