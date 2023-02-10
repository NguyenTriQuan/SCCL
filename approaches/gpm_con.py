# from cProfile import label
import sys, time, os
import math

import numpy as np
# from pytest import param
# from sqlalchemy import false
# from sympy import arg
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import kornia as K

import time
import csv
from utils import *
import networks.gpm_con_net as network
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

class Appr(object):

    def __init__(self,inputsize=None,taskcla=None,args=None,thres=1e-3,lamb='0',nepochs=100,sbatch=256,val_sbatch=256,
                lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=10,optim='Adam',tasknum=1,fix=False,norm_type=None):
        Net = getattr(network, args.arch)
        self.model = Net(input_size=inputsize, norm_type=args.norm_type).to(device)
        self.ncla = self.model.ncla
        
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
        self.lr_score = args.lr_score
        self.lr_rho = args.lr_rho
        self.lr_min = lr/100
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience 
        self.clipgrad = clipgrad
        self.optim = args.optimizer
        self.tasknum = args.tasknum
        self.fix = args.fix
        self.experiment = args.experiment
        self.approach = args.approach
        self.arch = args.arch
        self.seed = args.seed
        self.norm_type = args.norm_type
        self.ablation = args.ablation
        self.logger = None
        self.thres = args.thres
        self.prune_method = args.prune_method
        self.temperature = args.temperature
        self.contrast_mode = 'all'
        self.base_temperature = 0.07

        self.args = args
        self.lamb = float(args.lamb)
        self.check_point = None
        
        self.get_name()
        self.model.mem_images = torch.empty(0)
        self.model.mem_targets = torch.empty(0, dtype=int)

    def get_name(self):
        self.log_name = '{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_optim_{}_norm_{}'.format(
                                        self.experiment, self.approach, self.ablation, self.arch, self.seed, self.lamb, 
                                        self.lr, self.batch_size, self.nepochs, self.optim, self.norm_type)
        
    def resume(self):
        try:
            self.get_name()
            self.check_point = torch.load(f'../result_data/trained_model/{self.log_name}.model')
            self.model = self.check_point['model']
            self.ncla = self.model.ncla
            self.cur_task = len(self.ncla)-2
            return self.cur_task
        except:
            return 0

    def count_params(self):
        return self.model.count_params()[0]

    def _get_optimizer(self, lr=None):
        if lr is None: lr=self.lr
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        return optimizer

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

        if self.check_point is None:
            print('Training new task')
            self.model.ncla.append(self.model.ncla[-1]+ncla)
            self.ncla = self.model.ncla
            print(self.ncla)
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
            self.get_name()
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
        else: 
            print('Continue training current task')

        self.model = self.model.to(device)
        self.ncla = self.model.ncla
        self.cur_task = len(self.ncla)-2
        self.threshold = np.array([0.99] * 5) + t*np.array([0.003] * 5)
        print(self.log_name)

        self.mean = train_loader.dataset.tensors[0].mean(dim=(0, 2, 3))
        var = train_loader.dataset.tensors[0].var(dim=(0, 2, 3))
        next_ks = self.model.DM[0].ks
        self.std = (var.sum() * next_ks) ** 0.5
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform)
        self.updateGPM(train_loader, valid_transform, self.threshold)
        self.check_point['model'] = self.model
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
        self.check_point = None  


    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform):
        self.get_classes_statistic(t, train_loader, valid_transform)

        train_loss,train_acc=self.eval(t, train_loader, valid_transform)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')
        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        self.nepochs = self.args.nepochs

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        self.optimizer = self.check_point['optimizer']
        start_epoch = self.check_point['epoch'] + 1
        squeeze = self.check_point['squeeze']

        train_accs = []
        valid_accs = []
        best_acc = valid_acc

        for e in range(start_epoch, self.nepochs):
            clock0=time.time()
            train_loss = self.train_epoch(t, train_loader, train_transform, valid_transform)
            clock1=time.time()
            self.get_classes_statistic(t, train_loader, valid_transform)
            _,train_acc=self.eval(t, train_loader, valid_transform)
            clock2=time.time()
            print('| Epoch {:2d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                e+1,1000*(clock1-clock0),
                1000*(clock2-clock1),train_loss,100*train_acc),end='')

            valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
            print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                print(' *', end='')
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                        
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)

            print()
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        self.get_classes_statistic(t, train_loader, valid_transform)

    def train_batch(self, t, images, targets):
        outputs = self.model.forward(images)
        outputs = F.normalize(outputs, dim=1)
        loss = self.sup_con_loss(outputs, targets) 
        # if self.cur_task > 0:
        #     old_features_mean = self.model.features_mean[:self.ncla[-2]]
        #     old_features_mean = F.normalize(old_features_mean, dim=1)
        #     loss += self.lamb * self.old_con_loss(outputs, old_features_mean)
        self.optimizer.zero_grad()
        loss.backward() 
        self.model.project_gradient()
        self.optimizer.step()
        # self.model.normalize()
        return loss.detach().cpu().item()

    def eval_batch(self, t, images, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(images)
            features_mean = self.model.features_mean[self.ncla[t]: self.ncla[t+1]]
            # features_mean = self.model.features_mean
            outputs = F.normalize(outputs, dim=1)
            features_mean = F.normalize(features_mean, dim=1)
            predicts = torch.matmul(outputs, features_mean.T)
            targets = targets - self.ncla[t]

        values,indices=predicts.max(1)
        hits=(indices==targets).float()
        return hits.sum().data.cpu().numpy()

    def train_epoch(self, t, data_loader, train_transform, valid_transform):
        self.model.train()
        total_loss = 0
        total_num = 0
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = torch.cat([valid_transform(images), train_transform(images)], dim=0)
                # images = (images - self.mean.view(1, -1, 1, 1)) / self.std
                # images = torch.cat([images, images], dim=0)
                targets = torch.cat([targets, targets], dim=0)
            total_loss += self.train_batch(t, images, targets)
            total_num += len(targets)
        return total_loss

    def eval(self, t, data_loader, valid_transform):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
                # images = (images - self.mean.view(1, -1, 1, 1)) / self.std
                    
            hits = self.eval_batch(t, images, targets)
            total_acc += hits
            total_num += len(targets)
                
        return 0, total_acc/total_num

    def get_classes_statistic(self, t, data_loader, valid_transform):
        self.model.eval()
        features = []
        labels = []
        for images, targets in data_loader:
            images = images.to(device)
            if valid_transform:
                images = valid_transform(images)
            
            outputs = self.model.forward(images)
            features.append(outputs.detach())
            labels.append(targets)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        features_mean = []
        features_var = []
        begin = self.ncla[-2]
        end = self.ncla[-1]
        for cla in range(begin, end):
            ids = (labels == cla)
            cla_features = features[ids]
            features_mean.append(cla_features.mean(0))
            features_var.append(cla_features.var(0))

        features_mean = torch.stack(features_mean, dim=0).to(device)
        features_var = torch.stack(features_var, dim=0).to(device)
        if self.model.features_mean is None:
            self.model.features_mean = features_mean # [num classes, feature dim]
            self.model.features_var = features_var
        else:
            self.model.features_mean = torch.cat([self.model.features_mean[:begin], features_mean], dim=0)
            self.model.features_var = torch.cat([self.model.features_var[:begin], features_var], dim=0)

    def updateGPM(self, data_loader, valid_transform, threshold): 
        # Collect activations by forward pass
        self.val_batch_size = 125
        batch_list=[2*12,100,100,125,125]
        inputs = []
        N = 0
        for i, (images, targets) in enumerate(data_loader):
            inputs.append(images)
            N += images.shape[0]
            if N >= self.val_batch_size:
                break
        self.model.eval()
        self.model.track_input(True)
        inputs = torch.cat(inputs, dim=0).to(device)[:self.val_batch_size]
        if valid_transform:
            images = valid_transform(images)
        outputs = self.model(inputs)
        for i, m in enumerate(self.model.DMS):
            self.model.DM[i].act = self.model.DM[i].act[: batch_list[i]]
        self.model.get_feature(threshold)
        self.model.track_input(False)
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i, m in enumerate(self.model.DM):
            if m.feature is not None:
                print ('Layer {} : {}/{}'.format(i+1, m.feature.shape[1], m.feature.shape[0]))
        print('-'*40)

    def sup_con_loss(self, features, labels):
        sim = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(features.shape[0]).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)        
        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss

    def old_con_loss(self, features, old_features_mean):
        sim = torch.div(
            torch.matmul(features, old_features_mean.T),
            self.temperature)

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        loss = torch.log(exp_logits.sum(1))
        return loss.mean()

    def SupConLoss(self, features, labels=None, mask=None):
        
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

