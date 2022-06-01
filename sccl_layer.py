import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal, Normal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor
from typing import Optional, Any
from torch.nn import init
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple, Union


normal = Normal(0, 1)

class _DynamicLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, batch_norm=False, dropout=0.0):
        super(_DynamicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0))
        else:
            self.register_parameter("bias", None)

        self.shape_in = [in_features]
        self.shape_out = [out_features]


        if batch_norm:
            self.batch_norm = nn.ModuleList([DynamicBatchNorm(self.out_features)])
        else:
            self.batch_norm = None

        self.weight_pre = [None]
        self.bias_pre = [None]
        self.mask_pre = [None]
        self.grad_in = 0
        self.grad_out = 0

        self.norm = [0]

    def restrict_gradients(self):
        self.weight.grad.data[:self.shape_out[-2]][:, :self.shape_in[-2]] = 0
        if self.bias is not None:
            self.bias.grad.data[:self.shape_out[-2]] = 0

        # self.weight.grad.data[:self.shape_out[-2]][:, self.shape_in[-2]:] = 0


    def squeeze(self, mask_in, mask_out):
        self.weight.data = self.weight.data[mask_out][:,mask_in].clone()
        # self.weight_pre[-1] = self.weight_pre[-1][mask_out][:,mask_in].clone()
        self.weight.grad = None

        if self.bias is not None:
            self.bias.data = self.bias.data[mask_out].clone()
            # self.bias_pre[-1] = self.bias_pre[-1][mask_out].clone()
            self.bias.grad = None

        self.in_features, self.out_features = self.weight.shape[1], self.weight.shape[0]
        self.shape_in[-1] = self.in_features
        self.shape_out[-1] = self.out_features

        if self.batch_norm is not None:
            self.batch_norm[-1].squeeze(mask_out)

    # def forget(self, mask_in, mask_out):
    #   # print(self.shape_out, self.shape_in)
    #   for t in range(1, len(self.shape_out)):
    #       self.shape_out[t] = mask_out[:self.shape_out[t]].sum().item()
    #       self.shape_in[t] = mask_in[:self.shape_in[t]].sum().item()

    #   # print(self.shape_out, self.shape_in)
    #   self.squeeze(mask_in, mask_out)

    def expand(self, add_in=0, add_out=0):

        if isinstance(self, DynamicLinear):
            weight = torch.Tensor(self.out_features+add_out, self.in_features+add_in).cuda()
        else:
            weight = torch.Tensor(self.out_features+add_out, (self.in_features+add_in) // self.groups, *self.kernel_size).cuda()

        # init weight
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # print(self.out_features, self.in_features, add_out, add_in)
        if add_out != 0 and self.in_features != 0:
            nn.init.kaiming_uniform_(weight[self.out_features:self.out_features+add_out][:, :self.in_features], a=math.sqrt(5))

        if add_in != 0 and self.out_features != 0:
            nn.init.kaiming_uniform_(weight[:self.out_features][:, self.in_features:self.in_features+add_in], a=math.sqrt(5))
        
        if add_in != 0 and add_out != 0:
            nn.init.kaiming_uniform_(weight[self.out_features:self.out_features+add_out][:, self.in_features:self.in_features+add_in], a=math.sqrt(5))

        # nn.init.normal_(weight, 0, 0.1)
        weight[:self.out_features][:, :self.in_features] = self.weight.data.clone()
        # weight[:self.out_features][:, self.in_features:] = 0
        self.weight = nn.Parameter(weight)
        # self.weight_pre.append(torch.ones(self.weight.shape).cuda().bool())

        if self.bias is not None:
            bias = torch.Tensor(self.out_features+add_out).uniform_(0,0).cuda()
            bias[:self.out_features] = self.bias.data.clone()
            self.bias = nn.Parameter(bias)
            # self.bias_pre.append(torch.ones(self.bias.shape).cuda().bool())
        
        self.in_features += add_in
        self.out_features += add_out

        self.shape_in[-1] = self.in_features
        self.shape_out[-1] = self.out_features

        if self.batch_norm is not None:
            self.batch_norm.append(DynamicBatchNorm(self.out_features))

        self.strength_in = np.prod(self.weight[self.shape_out[-2]:].shape)
        self.strength_out = np.prod(self.weight[:, self.shape_in[-2]:].shape)

        self.mask_pre.append(torch.ones(self.shape_out[-2]).cuda().bool())

        # if self.weight_pre[-1].numel() == 0:
        #   self.weight_pre[-1] = 1
        #   self.bias_pre[-1] = 1

        # print(self.weight_pre)


    def new_task(self):
        self.shape_in.append(self.in_features)
        self.shape_out.append(self.out_features)
        self.grad_in = 0
        self.grad_out = 0
        self.grad_weight = 0
        self.grad_bias = 0

    # def set_cur_task(self, t):
    #   self.cur_weight = self.weight[]
    #   self.cur_bias = self.bias

    def bn_norm(self):
        if self.batch_norm is not None:
            return (self.batch_norm[-1].weight[self.shape_out[-2]:]**2 + 
                self.batch_norm[-1].bias[self.shape_out[-2]:]**2) ** (1/2)
        else: 
            return 1

    def backward_reg(self, t):
        backward_weights = self.weight[:self.shape_out[t]][:, self.shape_in[t]:]
        return backward_weights.norm(2)

    def num_add(self, add_in, add_out, t=-1):
        if isinstance(self, DynamicLinear):
            count = (self.shape_out[t]+add_out) * (self.shape_in[t]+add_in)
        else:
            count = (self.shape_out[t]+add_out) * ((self.shape_in[t]+add_in)//self.groups) * np.prod(self.kernel_size)

        count -= np.prod(self.weight[:self.shape_out[t]][:, :self.shape_in[t]].shape)
        if self.bias is not None:
            count += add_out
        if self.batch_norm is not None:
            count += (self.shape_out[t]+add_out)*2

        return count

    def add_out_factor(self, add_in, t=-1):
        if isinstance(self, DynamicLinear):
            in_factor = (self.shape_in[t]+add_in)
        else:
            in_factor = (self.shape_in[t]+add_in) * np.prod(self.kernel_size) // self.groups

        bias_factor = 0 if self.bias is None else 1
        bn_factor = 0 if self.batch_norm is None else 2

        # add_params = add_out*(in_factor+bias_factor+bn_factor) + self.shape_out[t]*(in_factor+bn_factor) - np.prod(self.weight[:self.shape_out[t]][:, :self.shape_in[t]].shape)
        # add_params = a*add_out + b
        a = (in_factor+bias_factor+bn_factor)
        b = self.shape_out[t]*(in_factor+bn_factor) - np.prod(self.weight[:self.shape_out[t]][:, :self.shape_in[t]].shape)

        return a, b

    def add_in_factor(self, add_out, t=-1):
        if isinstance(self, DynamicLinear):
            out_factor = (self.shape_out[t]+add_out)
        else:
            out_factor = (self.shape_out[t]+add_out) * np.prod(self.kernel_size) // self.groups

        bias_factor = 0 if self.bias is None else 1
        bn_factor = 0 if self.batch_norm is None else 2
        # add_params = (self.shape_in[t]+add_in) * out_factor + add_out*(bias_factor+bn_factor) + self.shape_out[t]*bn_factor - np.prod(self.weight[:self.shape_out[t]][:, :self.shape_in[t]].shape)
        # add_params = a*add_in + b
        a = out_factor
        b = self.shape_in[t]*out_factor + add_out*(bias_factor+bn_factor) + self.shape_out[t]*bn_factor - np.prod(self.weight[:self.shape_out[t]][:, :self.shape_in[t]].shape)

        return a, b


class DynamicLinear(_DynamicLayer):

    def __init__(self, in_features, out_features, bias=True, batch_norm=False, dropout=0.0):
        super(DynamicLinear, self).__init__(in_features, out_features, bias, batch_norm, dropout)
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if out_features != 0 and in_features != 0:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        
    def forward(self, x, t):
        weight = self.weight[:self.shape_out[t]][:, :self.shape_in[t]] #* self.weight_pre[t]
        bias = (self.bias[:self.shape_out[t]] if self.bias is not None else None) #* self.bias_pre[t]

        # try:
        #   weight = (weight /  self.norm[t].view(-1, 1)).clone()
        #   bias = (bias / self.norm[t]).clone()
        # except:
        #   pass

        # if self.shape_in[t-1] == self.shape_in[t]:
        #   weight = self.weight[self.shape_out[t-1]:self.shape_out[t]][:, :self.shape_in[t]]
        # else:
        #   weight = self.weight[self.shape_out[t-1]:self.shape_out[t]][:, self.shape_in[t-1]:self.shape_in[t]]
        # bias = self.bias[self.shape_out[t-1]:self.shape_out[t]] if self.bias is not None else None
        # print(weight.shape)
        # if weight.numel() == 0:
        #   return 0

        output = F.linear(x, weight, bias)
        # output = F.linear(x, self.cur_weight, self.cur_bias)

        if self.batch_norm is not None:
            output = self.batch_norm[t](output)

        output[:, :self.shape_out[t-1]] = output[:, :self.shape_out[t-1]] * self.mask_pre[t]
        
        return output

    def norm_in(self):
        norm = self.weight[self.shape_out[-2]:].norm(2, dim=1)
        if self.bias is not None:
            norm = (norm**2 + self.bias[self.shape_out[-2]:]**2)**(1/2)

        # norm = self.weight.norm(2, dim=1)
        # if self.bias is not None:
        #   norm = (norm**2 + self.bias**2)**(1/2)

        return norm

    def norm_out(self, size=None):
        if size is None:
            norm = self.weight[:, self.shape_in[-2]:].norm(2, dim=0)
        else:
            norm = self.weight[:, self.shape_in[-2]:].view(size).norm(2, dim=(0,2,3))

        # if size is None:
        #   norm = self.weight.norm(2, dim=0)
        # else:
        #   norm = self.weight.view(size).norm(2, dim=(0,2,3))

        return norm

    def norm_in_grad(self):
        S = (self.weight.data*self.weight.grad.data).sum(dim=1) + self.bias.data*self.bias.grad.data
        return S

    def norm_out_grad(self, size=None):
        if size is None:
            S = (self.weight.data*self.weight.grad.data).sum(dim=0)
        else:
            S = (self.weight.data.view(size)*self.weight.grad.data.view(size)).sum(dim=(0,2,3))
        return S

class DynamicClassifier(DynamicLinear):
    """docstring for DynamicClassifier"""
    def __init__(self, in_features, out_features, bias=True):
        super(DynamicClassifier, self).__init__(in_features, out_features, bias,)

        
    def forward(self, x, t):
        weight = self.weight[:self.shape_out[t]][:, :self.shape_in[t]]
        # weight = self.weight[:self.shape_out[t]][:, self.shape_in[t-1]:self.shape_in[t]]

        bias = self.bias[:self.shape_out[t]] if self.bias is not None else None
        # print(weight.shape)
        output = F.linear(x, weight, bias)

        if self.batch_norm is not None:
            output = self.batch_norm[t](output)

        output[:, :self.shape_out[t-1]] = output[:, :self.shape_out[t-1]] * self.mask_pre[t]
        return output
        
            
        
class _DynamicConvNd(_DynamicLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, bias, batch_norm):
        super(_DynamicConvNd, self).__init__(in_features, out_features, bias, batch_norm)
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
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features // groups, *kernel_size))
        if out_features != 0 and in_features != 0:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class DynamicConv2D(_DynamicConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DynamicConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, bias, batch_norm)
    
    def forward(self, x, t):

        weight = self.weight[:self.shape_out[t]][:, :self.shape_in[t]] #* self.weight_pre[t]
        bias = (self.bias[:self.shape_out[t]] if self.bias is not None else None) #* self.bias_pre[t]

        # if self.shape_in[t-1] == self.shape_in[t]:
        #   weight = self.weight[self.shape_out[t-1]:self.shape_out[t]][:, :self.shape_in[t]]
        # else:
        #   weight = self.weight[self.shape_out[t-1]:self.shape_out[t]][:, self.shape_in[t-1]:self.shape_in[t]]

        # bias = self.bias[self.shape_out[t-1]:self.shape_out[t]] if self.bias is not None else None

        output = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        # output = F.conv2d(x, self.cur_weight, self.cur_bias, self.stride, self.padding, self.dilation, self.groups)
        if self.batch_norm is not None:
            output = self.batch_norm[t](output)

        output[:, :self.shape_out[t-1]] = output[:, :self.shape_out[t-1]] * self.mask_pre[t].view(1,-1,1,1)
        return output


    def norm_in(self):
        norm = self.weight[self.shape_out[-2]:].norm(2, dim=(1,2,3))
        if self.bias is not None:
            norm = (norm**2 + self.bias[self.shape_out[-2]:]**2)**(1/2)

        # norm = self.weight.norm(2, dim=(1,2,3))
        # if self.bias is not None:
        #   norm = (norm**2 + self.bias**2)**(1/2)

        return norm

    def norm_out(self):
        norm = self.weight[:, self.shape_in[-2]:].norm(2, dim=(0,2,3))

        # norm = self.weight.norm(2, dim=(0,2,3))

        return norm

    def norm_in_grad(self):
        S = (self.weight.data*self.weight.grad.data).sum(dim=(1,2,3)) + self.bias.data*self.bias.grad.data
        return S

    def norm_out_grad(self):
        S = (self.weight.data*self.weight.grad.data).sum(dim=(0,2,3))
        return S


class _DynamicConvTransposeNd(_DynamicConvNd):
    def __init__(self, in_features, out_features, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, batch_norm) -> None:
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        super(_DynamicConvNd, self).__init__(
            in_features, out_features, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, batch_norm)

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(self, input, output_size,
                        stride, padding, kernel_size,
                        dilation):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class DynamicConvTranspose2D(_DynamicConvTransposeNd):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        batch_norm = None
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(_DynamicConvTransposeNd, self).__init__(
            in_features, out_features, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, batch_norm)

    def forward(self, input, t, output_size = None):

        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        weight = self.weight[:self.shape_out[t]][:, :self.shape_in[t]]
        bias = self.bias[:self.shape_out[t]] if self.bias is not None else None

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]

        output = F.conv_transpose2d(
            input,weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

        if self.batch_norm is not None:
            output = self.batch_norm[t](output)

        return output


class DynamicBatchNorm(nn.BatchNorm2d):
    """docstring for DynamicBatchNorm"""

    def squeeze(self, mask):

        self.weight.data = self.weight.data[mask].clone()
        self.bias.data = self.bias.data[mask].clone()

        self.weight.grad = None
        self.bias.grad = None

        if self.track_running_stats:
            self.running_mean = self.running_mean[mask].clone()
            self.running_var = self.running_var[mask].clone()

        self.num_features = self.weight.shape[0]

    def _check_input_dim(self, input):
        return


# class _DynamicNormBase(nn.Module):

#   _version = 2
#   __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
#   num_features: int
#   eps: float
#   momentum: float
#   affine: bool
#   track_running_stats: bool
#   # WARNING: weight and bias purposely not defined here.
#   # See https://github.com/pytorch/pytorch/issues/39670

#   def __init__(
#       self,
#       num_features: int,
#       eps: float = 1e-5,
#       momentum: float = 0.1,
#       affine: bool = True,
#       track_running_stats: bool = True,
#       device=None,
#       dtype=None
#   ) -> None:
#       factory_kwargs = {'device': device, 'dtype': dtype}
#       super(_DynamicNormBase, self).__init__()
#       self.num_features = num_features
#       self.eps = eps
#       self.momentum = momentum
#       self.affine = affine
#       self.track_running_stats = track_running_stats
#       if self.affine:
#           self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
#           self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
#       else:
#           self.register_parameter("weight", None)
#           self.register_parameter("bias", None)
#       if self.track_running_stats:
#           self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
#           self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
#           self.running_mean: Optional[Tensor]
#           self.running_var: Optional[Tensor]
#           self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long,**{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
#       else:
#           self.register_buffer("running_mean", None)
#           self.register_buffer("running_var", None)
#           self.register_buffer("num_batches_tracked", None)
#       self.reset_parameters()

#       self.shape = [0]

#   def reset_running_stats(self) -> None:
#       if self.track_running_stats:
#           # running_mean/running_var/num_batches... are registered at runtime depending
#           # if self.track_running_stats is on
#           self.running_mean.zero_()  # type: ignore[union-attr]
#           self.running_var.fill_(1)  # type: ignore[union-attr]
#           self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

#   def reset_parameters(self) -> None:
#       self.reset_running_stats()
#       if self.affine:
#           init.ones_(self.weight)
#           init.zeros_(self.bias)

#   def _check_input_dim(self, input):
#       raise NotImplementedError

#   def extra_repr(self):
#       return (
#           "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
#           "track_running_stats={track_running_stats}".format(**self.__dict__)
#       )

#   def _load_from_state_dict(
#       self,
#       state_dict,
#       prefix,
#       local_metadata,
#       strict,
#       missing_keys,
#       unexpected_keys,
#       error_msgs,
#   ):
#       version = local_metadata.get("version", None)

#       if (version is None or version < 2) and self.track_running_stats:
#           # at version 2: added num_batches_tracked buffer
#           #               this should have a default value of 0
#           num_batches_tracked_key = prefix + "num_batches_tracked"
#           if num_batches_tracked_key not in state_dict: 
#               state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

#       super(_NormBase, self)._load_from_state_dict(
#           state_dict,
#           prefix,
#           local_metadata,
#           strict,
#           missing_keys,
#           unexpected_keys,
#           error_msgs,
#       )

#   def restrict_gradients(self):
#       self.weight.grad.data[:self.shape[-2]] = 0
#       self.bias.grad.data[:self.shape[-2]] = 0

#   def squeeze(self, mask):
#       if len(self.shape) > 1:
#           mask[:self.shape[-2]] = True

#       self.weight.data = self.weight.data[mask].clone()
#       self.bias.data = self.bias.data[mask].clone()

#       self.weight.grad = None
#       self.bias.grad = None

#       if self.track_running_stats:
#           self.running_mean = self.running_mean[mask].clone()
#           self.running_var = self.running_var[mask].clone()

#       self.num_features = self.weight.shape[0]
#       self.shape[-1] = self.num_features

#   def expand(self, add):
#       weight = torch.Tensor(self.num_features+add).cuda()
#       init.ones_(weight)
#       weight[:self.num_features] = self.weight.data.clone()
#       self.weight = nn.Parameter(weight)

#       bias = torch.Tensor(self.num_features+add).cuda()
#       init.zeros_(bias)
#       bias[:self.num_features] = self.bias.data.clone()
#       self.bias = nn.Parameter(bias)

#       if self.track_running_stats:
#           running_mean = torch.ones(self.num_features+add).cuda()
#           running_mean[:self.num_features] = self.running_mean.clone()
#           self.running_mean = running_mean

#           running_var = torch.zeros(self.num_features+add).cuda()
#           running_var[:self.num_features] = self.running_var.clone()
#           self.running_var = running_var

#       self.num_features += add
#       self.shape[-1] = self.num_features
        
#   def new_task(self):
#       self.shape.append(self.num_features)


# class DynamicBatchNorm(_DynamicNormBase):
#   def __init__(
#       self,
#       num_features,
#       eps=1e-5,
#       momentum=0.1,
#       affine=True,
#       track_running_stats=True,
#       device=None,
#       dtype=None
#   ):
#       factory_kwargs = {'device': device, 'dtype': dtype}
#       super(DynamicBatchNorm, self).__init__(
#           num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
#       )

#   def forward(self, input: Tensor, t) -> Tensor:
#       # self._check_input_dim(input)

#       # exponential_average_factor is set to self.momentum
#       # (when it is available) only so that it gets updated
#       # in ONNX graph when this node is exported to ONNX.
#       if self.momentum is None:
#           exponential_average_factor = 0.0
#       else:
#           exponential_average_factor = self.momentum

#       if self.training and self.track_running_stats:
#           # TODO: if statement only here to tell the jit to skip emitting this when it is None
#           if self.num_batches_tracked is not None:  # type: ignore[has-type]
#               self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
#               if self.momentum is None:  # use cumulative moving average
#                   exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#               else:  # use exponential moving average
#                   exponential_average_factor = self.momentum

#       r"""
#       Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#       Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#       """
#       if self.training:
#           bn_training = True
#       else:
#           bn_training = (self.running_mean is None) and (self.running_var is None)

#       r"""
#       Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#       passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#       used for normalization (i.e. in eval mode when buffers are not None).
#       """
#       running_mean = self.running_mean[:self.shape[t]] if self.running_mean is not None else None
#       running_var = self.running_var[:self.shape[t]] if self.running_var is not None else None
#       return F.batch_norm(
#           input,
#           # If buffers are not to be tracked, ensure that they won't be updated
#           running_mean if not self.training or self.track_running_stats else None,
#           running_var if not self.training or self.track_running_stats else None,
#           self.weight[:self.shape[t]],
#           self.bias[:self.shape[t]],
#           bn_training,
#           exponential_average_factor,
#           self.eps,
#       )

if __name__ == '__main__':
    m = DynamicConvTranspose2D(1,2,3)
    m.new_task()
    m.expand(1,1)
    print(m.weight.shape)