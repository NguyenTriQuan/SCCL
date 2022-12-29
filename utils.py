import os,sys
import numpy as np
import random
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from torch._six import inf
import pandas as pd
from PIL import Image
from sklearn.feature_extraction import image
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torchvision import models
import time
import torch.nn.functional as F
# from layers.sccl_gpm_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
# from torchvision.models.resnet import *

# resnet_model = models.resnet18(pretrained=True).cuda()
# feature_extractor = nn.Sequential(*list(resnet_model.children())[:-4])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entropy(x):
    return -torch.sum(x * torch.log(x+0.0001), dim=1)

def log_likelihood(x, mean, var):
    # log N(x | mean, var)
    # x [bs, feat_dim], mean, var [num_cla, feat_dim]
    out = -((x.unsqueeze(1) - mean.unsqueeze(0)) ** 2) / (2 * var) - var.log() / 2 #- math.log(math.sqrt(2 * math.pi))
    return out.mean(-1)

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def ensemble_outputs(outputs):
    """
        pre_outputs: with batch_size repeated to batch_size * ensemble_numbers
        bs:          real batch_size
    """
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    # outputs = pre_outputs.split(bs)
    ## with shape [bs, num_cls, num_member]
    # outputs = torch.stack(outputs, dim=-1)
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=-1)

    return log_outputs

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def weighted_ensemble(outputs, weights, temperature):
    """
        pre_outputs: with batch_size repeated to batch_size * ensemble_numbers
        bs:          real batch_size
    """
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    # outputs = pre_outputs.split(bs)
    ## with shape [bs, num_cls, num_member]
    # outputs = torch.stack(outputs, dim=-1)
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    output_max, _ = torch.max(outputs, dim=-1, keepdim=True)
    weights = F.softmax(-weights / temperature, dim=-1).unsqueeze(1)
    # print(weights.view(-1))
    log_outputs = output_max + torch.log(torch.mean((outputs - output_max).exp() * weights, dim=-1, keepdim=True))
    return log_outputs.squeeze(-1)

def ensemble_features(outputs):
    outputs = F.normalize(outputs, dim=-2).mean(dim=-1)
    return F.normalize(outputs, dim=-1)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

# def entropy(x, exp=1):
#     y = F.softmax(x, dim=1)
#     y = y.pow(exp)
#     y = y/y.sum(1).view(-1,1).expand_as(y)
#     return (-y*y.log()).sum(1)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def gs_cal(t, x, y, criterion, model, sbatch=20):
    
    param_R = {}
    # Init
    
    for name, param in model.named_parameters():
        if len(param.size()) <= 1:
            continue
        name = name.split('.')[:-1]
        name = '.'.join(name)
        param = param.view(param.size(0), -1)
        param_R['{}'.format(name)]=torch.zeros((param.size(0)))
    
    # Compute
    model.train()

    for i in range(0,x.size(0),sbatch):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=x[b]
        target=y[b]

        # Forward and backward
        outputs = model.forward(t, images, True)
        cnt = 0
        
        for idx, j in enumerate(model.act):
            j = torch.mean(j, dim=0)
            if len(j.size())>1:
                j = torch.mean(j.view(j.size(0), -1), dim = 1).abs()
            model.act[idx] = j
            
        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.act[cnt].abs().detach()*sbatch
            cnt+=1 

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key]=(param_R[key]/x.size(0))
    return param_R

class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, lr_rho=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, param_name=None, lr_scale=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.param_name = param_name
        self.lr_rho = lr_rho
        self.lr_scale = lr_scale
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                n = self.param_name[i]
                
                if 'rho' in self.param_name[i]:
                    step_size = self.lr_rho * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

#                 p.data.addcdiv_(-step_size, self.lr_scale[n] * exp_avg, denom)
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


########################################################################################################################
def crop(x, patch_size, mode = 'train'):
    cropped_image = []
    arr_len = len(x)
    if mode == 'train':
        for idx in range(arr_len):
            
            patch = image.extract_patches_2d(image = x[idx].data.cpu().numpy(),
                                            patch_size = (patch_size, patch_size), max_patches = 1)[0]
            
            # Random horizontal flipping
            if random.random() > 0.5:
                patch = np.fliplr(patch)
            # Random vertical flipping
            if random.random() > 0.5:
                patch = np.flipud(patch)
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)
    elif mode == 'valid' or mode == 'test':
        for idx in range(arr_len):
            patch = x[idx].data.cpu().numpy()
            H,W,C = patch.shape
            patch = patch[H//2-patch_size//2:H//2+patch_size//2, W//2-patch_size//2:W//2+patch_size//2,:]
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)
        
    image_tensor=torch.stack(cropped_image).view(-1,3,patch_size,patch_size).cuda()
    return image_tensor


def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
        # count += (p!=0).int().sum().item()
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.2f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################
def copy_model(model):
    for module_ in model.net:
        if isinstance(module_, nn.ModuleList):
            for linear_ in module_:
                linear_.clean()
        if isinstance(module_, nn.ReLU) or isinstance(module_, nn.Linear) or isinstance(module_, nn.Conv2d) or isinstance(module_, nn.MaxPool2d) or isinstance(module_, nn.Dropout):
            module_.clean()

    return deepcopy(model)
    
def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20, split = False, args=None):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=x[b]
        target=y[b]
        
        # Forward and backward
        model.zero_grad()
        if split:
            outputs = model.forward(t, images)
        else:
            outputs=model.forward(t, images)
        loss=criterion(t,outputs,target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/x.size(0)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

def clip_relevance_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)

    return total_norm

########################################################################################################################

class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))


def naive_lip(model: nn.Module, input_size, t, n_iter: int = 10, eps=1e-3, n_samples=256):
    lip_2 = -1
    lip_max = -1
    lip_max_2 = -1
    lip_2_max = -1
    for i in range(n_iter):
        x1 = torch.randn([n_samples] + list(input_size)).to(device)
        alpha = ((torch.rand([n_samples] + list(input_size)) * 2 - 1) * eps).to(device)

        y1, y2 = model(x1, t), model(x1 + alpha, t)
        beta = y2-y1

        denominator_2 = torch.linalg.vector_norm(alpha.view(n_samples, -1), ord=float(2), dim=1)
        numerator_2 = torch.linalg.vector_norm(beta.view(n_samples, -1), ord=float(2), dim=1)
        lip_2 = max(lip_2, torch.div(numerator_2, denominator_2).max().item())

        denominator_max = torch.linalg.vector_norm(alpha.view(n_samples, -1), ord=float('inf'), dim=1)
        numerator_max = torch.linalg.vector_norm(beta.view(n_samples, -1), ord=float('inf'), dim=1)
        lip_max = max(lip_max, torch.div(numerator_max, denominator_max).max().item())

        lip_max_2 = max(lip_max_2, torch.div(numerator_max, denominator_2).max().item())
        lip_2_max = max(lip_2_max, torch.div(numerator_2, denominator_max).max().item())

    return lip_2, lip_max, lip_max_2, lip_2_max

def KMeans(x, K=2, Niter=10, verbose=False, use_cuda=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )
    return cl, c
