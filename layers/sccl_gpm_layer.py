
from re import S
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
from gmm_torch.gmm import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.dropout = dropout
        if first_layer:
            self.base_in_features = 0
            self.in_features = in_features
        else:
            self.base_in_features = in_features
            self.in_features = 0

        if last_layer:
            self.base_out_features = 0
            self.out_features = out_features
        else:
            self.base_out_features = out_features
            self.out_features = 0
            
        # bias = False # Alert fix later
        if bias:
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features))])
        else:
            self.register_parameter("bias", None)

        self.scale = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features))])

        self.shape_in = [self.in_features]
        self.shape_out = [self.out_features]

        self.norm_type = norm_type

        if norm_type:
            self.norm_layer = DynamicNorm(self.out_features, affine=True, track_running_stats=True, norm_type=norm_type)
        else:
            self.norm_layer = None

        self.next_layers = next_layers # where output of this layer go
        self.s = s
        self.mask = None
        
        self.projection_matrix = None
        self.feature = None 
        self.track_input = False
        self.sim = [None]
        self.cur_task = 0

    def forward(self, x, t):
        if self.track_input:
            self.act = x.detach()
        weight, bias = self.get_parameters(t)

        if weight.numel() == 0:
            return None

        if isinstance(self, DynamicLinear):
            output = F.linear(x, weight, bias)
            view = (1, -1)
        else:
            output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            view = (1, -1, 1, 1)

        if self.norm_layer is not None:
            output = self.norm_layer(output, t, self.dropout)

        # if self.mask is not None:
        #     output[:, self.shape_out[-2]:] = output[:, self.shape_out[-2]:] * self.mask.view(view)

        return output

    def norm_in(self, t=-1):
        if isinstance(self, DynamicLinear):
            norm_dim = (1)
        else:
            norm_dim = (1, 2, 3)
        weight = torch.cat([self.weight[t], self.fwt_weight[t]], dim=1)
        norm = weight.norm(2, dim=norm_dim)
        if self.bias is not None:
            norm = norm**2 + self.bias[t]**2     
        return norm ** 0.5

    def norm_out(self, t=-1):
        weight = torch.cat([self.next_layers[0].weight[t], self.next_layers[0].bwt_weight[t]], dim=0)
        if isinstance(self, DynamicLinear):
            norm_dim = (0)
        else:
            norm_dim = (0, 2, 3)
            if isinstance(self.next_layers[0], DynamicLinear):
                weight = weight.view(self.next_layers[0].weight[t].shape[0] + self.next_layers[0].bwt_weight[t].shape[0], 
                                    self.weight[t].shape[0], self.next_layers[0].s, self.next_layers[0].s)
        norm = weight.norm(2, dim=norm_dim)
        return norm

    def get_optim_params(self, ablation='full'):
        params = []
        if 'gpm' in ablation:
            params += [self.weight[-1], self.fwt_weight[-1], self.bwt_weight[-1]]
        else:
            for i in range(len(self.weight)):
                params += [self.weight[i], self.fwt_weight[i], self.bwt_weight[i]]
        # params += [self.scale[-1]]
        if self.bias:
            params += [self.bias[-1]]
        if self.norm_layer:
            if self.norm_layer.affine:
                params += [self.norm_layer.weight[-1], self.norm_layer.bias[-1]]
        return params

    def count_params(self, t):
        count = 0
        for i in range(t+1):
            count += self.weight[i].numel() + self.fwt_weight[i].numel() + self.bwt_weight[i].numel()
            if self.bias:
                count += self.bias[i].numel()
            if self.norm_layer:
                if self.norm_layer.affine:
                    count += self.norm_layer.weight[i].numel() + self.norm_layer.bias[i].numel()
        return count

    def get_reg(self):
        reg = 0
        strength = self.strength_in + self.next_layer[0].strength_out
        reg += self.norm_in().sum() * self.strength_in
        reg += self.norm_out().sum() * self.next_layer[0].strength_out
            
        if self.norm_layer:
            if self.norm_layer.affine:
                reg += self.norm_layer.reg().sum() * strength

        return reg, strength

    def get_importance(self):
        norm = self.norm_in() 
        norm *= self.norm_out()
        if self.norm_layer:
            if self.norm_layer.affine:
                norm *= self.norm_layer.reg()

        return norm

    def get_parameters(self, t):
        weight = torch.empty(0).to(device)
        if self.bias:
            bias = torch.empty(0).to(device)
        else:
            bias = None

        for i in range(1, t):
            weight = torch.cat([torch.cat([weight, self.fwt_weight[i]], dim=0), 
                                torch.cat([self.bwt_weight[i], self.weight[i]], dim=0)], dim=1)
            if self.bias:
                bias = torch.cat([bias, self.bias[i]])

        if self.training:
            if self.sim[t] is not None:
                if len(weight.shape) == 2:
                    weight *= torch.normal(1, self.sim[t]).view(-1, 1)
                else:
                    weight *= torch.normal(1, self.sim[t]).view(-1, 1, 1, 1)

        # if len(weight.shape) == 2:
        #     weight *= self.scale[-1].view(-1, 1)
        # elif len(weight.shape) == 4:
        #     weight *= self.scale[-1].view(-1, 1, 1, 1)

        weight = torch.cat([torch.cat([weight, self.fwt_weight[t]], dim=0), 
                            torch.cat([self.bwt_weight[t], self.weight[t]], dim=0)], dim=1)
        if self.bias:
            bias = torch.cat([bias, self.bias[t]])

        return weight, bias

    def get_mat(self):
        if isinstance(self, DynamicLinear):
            if self.feature is not None:
                padding = torch.zeros(self.shape_in[-1]-self.shape_in[-2], self.feature.shape[1]).to(device)
                self.feature = torch.cat([self.feature, padding], dim=0)
            return self.act.T
        else:
            if self.feature is not None:
                padding = torch.zeros(self.kernel_size[0]*self.kernel_size[1]*(self.shape_in[-1]-self.shape_in[-2]), self.feature.shape[1])
                self.feature = torch.cat([self.feature, padding.to(device)], dim=0)

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

    def get_feature(self, threshold):
        mat = self.get_mat()
        if self.feature is None:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            # criteria (Eq-5)
            sval_ratio = S**2
            sval_ratio = sval_ratio/sval_ratio.sum()

            r = (torch.cumsum(sval_ratio, dim=0) <= threshold).sum().item() #+1 
            self.feature = U[:, :r]
            print(sval_ratio[:r])

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
                if accumulated_sval <= threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                return
            # update GPM
            self.feature = torch.cat([self.feature, U[:, 0: r]], dim=1)  
            if self.feature.shape[1] > self.feature.shape[0] :
                self.feature = self.feature[:, 0: self.feature.shape[0]]

        self.projection_matrix = torch.mm(self.feature, self.feature.T)

    def project(self, params):
        sz =  params.size(0)
        return params - torch.mm(params.view(sz,-1), self.projection_matrix).view(params.size())

    def project_gradient(self, t):
        if self.projection_matrix is None:
            return

        weight_grad = torch.empty(0).to(device)
        for i in range(1, t):
            weight_grad = torch.cat([torch.cat([weight_grad, self.fwt_weight[i].grad], dim=0), 
                                torch.cat([self.bwt_weight[i].grad, self.weight[i].grad], dim=0)], dim=1)

        weight_grad = self.project(weight_grad)
        # temp = F.cosine_similarity(weight_grad.view(-1), weight_grad_projected.view(-1), dim=0)
        # print(temp.item(), math.acos(temp.item())/math.pi)
        for i in range(1, t): 
            self.weight[i].grad.data = weight_grad[self.shape_out[i-1]: self.shape_out[i]][:, self.shape_in[i-1]: self.shape_in[i]]
            self.fwt_weight[i].grad.data = weight_grad[self.shape_out[i-1]: self.shape_out[i]][:, : self.shape_in[i-1]]
            self.bwt_weight[i].grad.data = weight_grad[: self.shape_out[i-1]][:, self.shape_in[i-1]: self.shape_in[i]]

        # if self.fwt_weight[t].numel() != 0:
        #     self.fwt_weight[t].grad.data = self.project(self.fwt_weight[t].grad.data)

    def compute_project_similarity(self, t):
        if self.projection_matrix is None:
            return

        weight_grad = torch.empty(0).to(device)
        for i in range(1, t):
            weight_grad = torch.cat([torch.cat([weight_grad, self.fwt_weight[i].grad], dim=0), 
                                torch.cat([self.bwt_weight[i].grad, self.weight[i].grad], dim=0)], dim=1)

        weight_grad_update = self.project(weight_grad)
        size = weight_grad.shape[0]
        cos_sim = F.cosine_similarity(weight_grad.view(size, -1), weight_grad_update.view(size, -1), dim=1)
        ang_dis = torch.arccos(cos_sim.detach()) / math.pi
        p_dropout = ang_dis/2
        self.sim[-1] = 1 / (1 - p_dropout)
        # print(weight_grad.view(size, -1).norm(2, 1))
        # print(weight_grad_update.view(size, -1).norm(2, 1))
        print(ang_dis)

    def squeeze(self, optim_state):
        def apply_mask_out(param, mask_out):
            param.data = param.data[mask_out].clone()
            param.grad = None
            param_states = optim_state[param]
            for name, state in param_states.items():
                if isinstance(state, torch.Tensor):
                    param_states[name] = state[mask_out].clone()

        def apply_mask_in(param, mask_in):
            param.data = param.data[:, mask_in].clone()
            param.grad = None
            param_states = optim_state[param]
            for name, state in param_states.items():
                if isinstance(state, torch.Tensor):
                    param_states[name] = state[:, mask_in].clone()


        if self.mask is not None:
            mask_out = self.mask
            apply_mask_out(self.weight[-1], mask_out)
            apply_mask_out(self.fwt_weight[-1], mask_out)

            if self.bias:
                apply_mask_out(self.bias[-1], mask_out)

            self.out_features = self.shape_out[-2] + self.weight[-1].shape[0]
            self.shape_out[-1] = self.out_features

            if self.norm_layer:
                mask = torch.ones(self.norm_layer.shape[-2]).bool().to(device)
                mask = torch.cat([mask, mask_out])
                if self.norm_layer.affine:
                    apply_mask_out(self.norm_layer.weight[-1], mask)
                    apply_mask_out(self.norm_layer.bias[-1], mask)

                if self.norm_layer.track_running_stats:
                    self.norm_layer.running_mean[-1] = self.norm_layer.running_mean[-1][mask]
                    self.norm_layer.running_var[-1] = self.norm_layer.running_var[-1][mask]

                self.norm_layer.num_features = self.out_features
                self.norm_layer.shape[-1] = self.out_features

            for m in self.next_layers:
                if isinstance(m, DynamicLinear) and isinstance(self, DynamicConv2D):
                    mask_in = self.mask.view(-1,1,1).expand(self.mask.size(0),m.s,m.s).contiguous().view(-1)
                else:
                    mask_in = self.mask

                apply_mask_in(m.weight[-1], mask_in)
                apply_mask_in(m.bwt_weight[-1], mask_in)

                m.in_features = m.shape_in[-2] + m.weight[-1].shape[1]
                m.shape_in[-1] = m.in_features
            
        self.mask = None
        strength_in = self.weight[-1].data.numel() + self.fwt_weight[-1].data.numel()
        strength_out = self.next_layers[0].weight[-1].data.numel() + self.next_layers[0].bwt_weight[-1].data.numel()
        self.strength = (strength_in + strength_out)

    def expand(self, add_in=None, add_out=None, ablation='full', pre_scale=1):
        if add_in is None:
            add_in = self.base_in_features
        if add_out is None:
            add_out = self.base_out_features

        if isinstance(self, DynamicLinear):
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in).to(device)))
            # old neurons to new neurons
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features).to(device)))
            # new neurons to old neurons
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in).to(device)))
            fan_in = 1
        else:
            # new neurons to new neurons
            self.weight.append(nn.Parameter(torch.Tensor(add_out, add_in // self.groups, *self.kernel_size).to(device)))
            # old neurons to new neurons
            self.fwt_weight.append(nn.Parameter(torch.Tensor(add_out, self.in_features // self.groups, *self.kernel_size).to(device)))
            # new neurons to old neurons
            self.bwt_weight.append(nn.Parameter(torch.Tensor(self.out_features, add_in // self.groups, *self.kernel_size).to(device)))
            fan_in = np.prod(self.kernel_size)

        if self.bias:
            self.bias.append(nn.Parameter(torch.Tensor(add_out).uniform_(0, 0).to(device)))

        self.scale.append(nn.Parameter(torch.Tensor(self.out_features).uniform_(1, 1).to(device)))

        # if 'fwt' not in ablation or self.first_layer:
        #     fan_in *= (self.in_features + add_in)
        # else:
        #     fan_in *= add_in

        if self.first_layer:
            fan_in *= (self.in_features + add_in)
        else:
            fan_in *= add_in

        if fan_in != 0:
            # init
            gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            bound = gain * math.sqrt(3.0/fan_in)

            nn.init.uniform_(self.weight[-1], -bound, bound)
            nn.init.uniform_(self.bwt_weight[-1], -bound, bound)

            if 'fwt' not in ablation or self.first_layer:
                nn.init.uniform_(self.fwt_weight[-1], -bound, bound)
            else:
                nn.init.uniform_(self.fwt_weight[-1], 0, 0)

            # rescale old tasks params
            if 'scale' not in ablation:
                weight, bias = self.get_parameters(len(self.weight)-2)
                # mean = weight.mean()
                std = weight.std()
                bound_std = gain / math.sqrt(fan_in)
                scale = bound_std / std

                for i in range(self.cur_task):
                    self.weight[i].data *= scale
                    self.fwt_weight[i].data *= scale
                    self.bwt_weight[i].data *= scale
                    if self.bias:
                        self.bias[i].data *= pre_scale * scale
                    if self.norm_layer:
                        if self.norm_layer.track_running_stats:
                            for i in range(self.cur_task):
                                self.norm_layer.running_mean[i].data *= pre_scale * scale
                                self.norm_layer.running_var[i].data *= pre_scale * scale
        
                pre_scale *= scale
                

        # if self.projection_matrix is not None:
        #     self.fwt_weight[-1].data = self.project(self.fwt_weight[-1].data)

        # requires grad
        if 'gpm' in ablation:
            self.weight[-2].requires_grad_ = False
            self.fwt_weight[-2].requires_grad_ = False
            self.bwt_weight[-2].requires_grad_ = False
        if 'fwt' in ablation and not self.first_layer:
            self.fwt_weight[-1].requires_grad_ = False

        self.in_features += add_in
        self.out_features += add_out

        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)

        if self.norm_layer:
            self.norm_layer.expand(add_out)

        self.strength_in = self.weight[-1].data.numel() + self.fwt_weight[-1].data.numel()
        self.strength_out = self.weight[-1].data.numel() + self.bwt_weight[-1].data.numel()
        
        self.mask = None
        self.sim.append(None)
        self.cur_task += 1
        return pre_scale

    def proximal_gradient_descent(self, lr, lamb, total_strength):
        if isinstance(self, DynamicLinear):
            view_in = (-1, 1)
            view_out = (1, -1)
        else:
            view_in = (-1, 1, 1, 1)
            view_out = (1, -1, 1, 1)

        with torch.no_grad():
            strength = self.strength / total_strength
            # group lasso weights in
            norm = self.norm_in(-1)
            aux = 1 - lamb * lr * strength / norm
            aux = F.threshold(aux, 0, 0, False)
            self.weight[-1].data *= aux.view(view_in)
            self.fwt_weight[-1].data *= aux.view(view_in)
            if self.bias:
                self.bias[-1].data *= aux
            self.mask = (aux != 0)

            # group lasso weights out
            norm = self.norm_out(-1)
            aux = 1 - lamb * lr * strength / norm
            aux = F.threshold(aux, 0, 0, False)
            self.mask *= (aux != 0)
            if isinstance(self.next_layers[0], DynamicLinear) and isinstance(self, DynamicConv2D):
                aux = aux.view(-1,1,1).expand(aux.size(0),self.next_layers[0].s,self.next_layers[0].s).contiguous().view(-1)
                view_out = (1, -1)
            self.next_layers[0].weight[-1].data *= aux.view(view_out)
            self.next_layers[0].bwt_weight[-1].data *= aux.view(view_out)                    

            # group lasso affine weights
            if self.norm_layer:
                if self.norm_layer.affine:
                    norm = self.norm_layer.weight[-1][self.norm_layer.shape[-2]:]**2 + self.norm_layer.bias[-1][self.norm_layer.shape[-2]:]**2
                    norm = norm ** 0.5
                    aux = 1 - lamb * lr * strength / norm
                    aux = F.threshold(aux, 0, 0, False)
                    self.norm_layer.weight[-1].data[self.norm_layer.shape[-2]:] *= aux
                    self.norm_layer.bias[-1].data[self.norm_layer.shape[-2]:] *= aux
                    self.mask *= (aux != 0)
                    

class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(DynamicLinear, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)
        
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, self.in_features))])
        self.fwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(self.out_features, 0))])
        self.bwt_weight = nn.ParameterList([nn.Parameter(torch.Tensor(0, self.in_features))])
            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout):
        super(_DynamicConvNd, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)
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
                stride=1, padding=0, dilation=1, groups=1, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)



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
    
    def reg(self):
        return (self.weight[-1][self.shape[-2]:]**2 + self.bias[-1][self.shape[-2]:]**2) ** 1/2

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
        elif self.norm_type =='ln':
            output = self.layer_norm(input)
        elif self.norm_type =='bn_ln':
            # output = self.layer_norm(self.batch_norm(input, t) + input)
            output = self.batch_norm(input, t) + input
            # output = self.layer_norm(input) + self.batch_norm(input, t)

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
    
            