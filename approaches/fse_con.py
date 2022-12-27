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
import networks.fse_con_net as network
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
        self.lambs = [float(i) for i in args.lamb.split('_')]
        self.check_point = None
        
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)
        self.shape_out = self.model.DM[-1].shape_out
        self.cur_task = len(self.shape_out)-2
        self.ce = torch.nn.CrossEntropyLoss()

        self.get_name(self.tasknum-1)
        self.model.mem_images = torch.empty(0)
        self.model.mem_targets = torch.empty(0, dtype=int)

    def get_name(self, t):
        self.log_name = '{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_optim_{}_fix_{}_norm_{}_drop_{}'.format(
                                        self.experiment, self.approach, self.ablation, self.arch, self.seed,
                                                '_'.join([str(lamb) for lamb in self.lambs[:t+1]]), 
                                    self.lr, self.batch_size, self.nepochs, self.optim, self.fix, self.norm_type, self.args.ensemble_drop)
        
    def resume(self):
        for t in range(self.tasknum):
            try:
                self.get_name(t)

                self.check_point = torch.load(f'../result_data/trained_model/{self.log_name}.model')
                self.model = self.check_point['model']
                self.ncla = self.model.ncla
                self.shape_out = self.model.DM[-1].shape_out
                self.cur_task = len(self.shape_out)-2
                print('Resume from task', t)

                return t
            except:
                continue
        return 0

    def count_params(self):
        return self.model.count_params()[0]

    def _get_optimizer(self, lr=None):
        if lr is None: lr=self.lr

        params = self.model.get_optim_params()
        params = [{'params': params, 'lr':lr}]
        scores = [m.score for m in self.model.DM[:-1]]
        params += [{'params': scores, 'lr':self.lr_score}]
        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr,
                          weight_decay=0.0, momentum=0.9)
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

        if self.check_point is None:
            print('Training new task')
            self.model.expand(ncla, self.ablation)
            self.ncla = self.model.ncla
            self.model = self.model.to(device)
            self.shape_out = self.model.DM[-1].shape_out
            self.cur_task = len(self.shape_out)-2
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
            try:
                os.remove(f'../result_data/trained_model/{self.log_name}.model')
            except:
                pass
            self.get_name(t)
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
            self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
            self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=False, mask=True, mem=False)
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}

        else: 
            print('Continue training current task')

        # print(self.model.report())
        self.model = self.model.to(device)
        self.shape_out = self.model.DM[-1].shape_out
        self.cur_task = len(self.shape_out)-2

        self.lamb = sum(self.lambs[:self.cur_task+1])

        if self.experiment == 'mixture':
            self.factor = train_loader.dataset.tensors[0].shape[0]
            self.lamb = self.lamb / (self.factor**0.5)

        print('lambda', self.lamb)
        print(self.log_name)

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=True, mask=False, mem=False)
        if not self.check_point['squeeze']:
            self.check_point = None
            return 

        self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
        if 'phase2' not in self.ablation:
            self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=False, mask=False, mem=False)

        self.model.freeze(t)
        if 'scale' not in self.ablation:
            self.model.update_scale()
        # self.update_mem(train_loader, valid_transform)
        # self.model.get_mem_params()
        # mem_loader = DataLoader(TensorDataset(self.model.mem_images, self.model.mem_targets), batch_size=self.batch_size, shuffle=True)
        # self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        # self.train_phase(t+1, mem_loader, mem_loader, train_transform, valid_transform, squeeze=False, mask=False, mem=True)

        self.check_point = None  
        self.model.count_params()

        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask=True, over_param=False, mem=False)
        print(' Valid mask: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))
        # valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask=False, over_param=False, mem=True)
        # print(' Valid mem: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))   
        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask=True, over_param=True, mem=False)
        print(' Valid ensemble mask: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))    


    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform, squeeze, mask, mem):
        if mask or mem:
            over_param = False
        else:
            over_param = True
                    
        print(f'Train phase: mask: {mask}, over param: {over_param}, mem: {mem}')
        self.model.count_params()
        self.model.get_old_params(t)
        self.get_classes_statistic(t, train_loader, valid_transform, mask, over_param, mem)
        train_loss,train_acc=self.eval(t, train_loader, valid_transform, mask, over_param, mem)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask, over_param, mem)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        if mask:
            self.nepochs = 100
        elif mem:
            self.nepochs = 50
        else:
            self.nepochs = self.args.nepochs

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        self.optimizer = self.check_point['optimizer']
        start_epoch = self.check_point['epoch'] + 1
        squeeze = self.check_point['squeeze']

        train_accs = []
        valid_accs = []
        if squeeze:
            best_acc = train_acc
        else:
            best_acc = valid_acc

        if True:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(t, train_loader, train_transform, valid_transform, squeeze, lr, mask, mem)
                clock1=time.time()
                self.get_classes_statistic(t, train_loader, valid_transform, mask, over_param, mem)
                train_loss,train_acc=self.eval(t, train_loader, valid_transform, mask, over_param, mem)
                clock2=time.time()
                print('| Epoch {:2d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask, over_param, mem)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if squeeze:
                    self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                    torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                    # model_count, layers_count = self.model.count_params()
                    # if self.logger is not None:
                    #     self.logger.log_metric('num params', model_count, epoch=e)
                elif mask or mem:
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                        print(' *', end='')
                else:
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
                # if self.logger is not None:
                #     self.logger.log_metrics({
                #         'train acc':train_acc,
                #         'valid acc':valid_acc
                #     }, epoch=e)

        # except KeyboardInterrupt:
        #     print('KeyboardInterrupt')
        #     self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        #     self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        self.get_classes_statistic(t, train_loader, valid_transform, mask, over_param, mem)
        # print(train_accs)
        # print(valid_accs)

    def train_batch(self, t, images, targets, squeeze, lr, mask, mem):
        outputs = self.model.forward(images, t, mask, mem)
        outputs = F.normalize(outputs, dim=1)
        loss = self.sup_con_cl_loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
        if squeeze:
            self.model.proximal_gradient_descent(lr, self.lamb)

    def eval_batch(self, t, images, targets, mask=True, over_param=False, mem=True):
        self.model.eval()
        with torch.no_grad():
            if mem:
                self.model.get_old_params(self.cur_task+1)
                outputs_mem = self.model.forward(images, self.cur_task+1, mask=True, mem=True)
            if t is None:
                sims = []
                for i in range(self.cur_task+1):
                    self.model.get_old_params(i)
                    sim = 0
                    if mask:
                        outputs = self.model.forward(images, i, mask=True, mem=False)
                        features_mean = self.model.features_mean_mask[self.ncla[i]:self.ncla[i+1]]
                        outputs = F.normalize(outputs, dim=1)
                        features_mean = F.normalize(features_mean, dim=1)
                        sim += torch.matmul(outputs, features_mean.T)
                    if over_param:
                        outputs = self.model.forward(images, i, mask=False, mem=False)
                        features_mean = self.model.features_mean[self.ncla[i]:self.ncla[i+1]]
                        outputs = F.normalize(outputs, dim=1)
                        features_mean = F.normalize(features_mean, dim=1)
                        sim += torch.matmul(outputs, features_mean.T)

                    sims.append(sim)
                predicts = torch.cat(sims, dim=1)
                if mem:
                    outputs = F.normalize(outputs_mem, dim=1)
                    features_mean = F.normalize(self.model.features_mean_mem, dim=1)
                    predicts += torch.matmul(outputs, features_mean.T)
            elif t <= self.cur_task:
                self.model.get_old_params(t)
                sim = 0
                if mask:
                    outputs = self.model.forward(images, t, mask=True, mem=False)
                    features_mean = self.model.features_mean_mask[self.ncla[t]:self.ncla[t+1]]
                    outputs = F.normalize(outputs, dim=1)
                    features_mean = F.normalize(features_mean, dim=1)
                    sim += torch.matmul(outputs, features_mean.T)
                if over_param:
                    outputs = self.model.forward(images, t, mask=False, mem=False)
                    features_mean = self.model.features_mean[self.ncla[t]:self.ncla[t+1]]
                    outputs = F.normalize(outputs, dim=1)
                    features_mean = F.normalize(features_mean, dim=1)
                    sim += torch.matmul(outputs, features_mean.T)
                if mem:
                    outputs = F.normalize(outputs_mem, dim=1)
                    features_mean = self.model.features_mean_mem[self.ncla[t]:self.ncla[t+1]]
                    features_mean = F.normalize(features_mean, dim=1)
                    sim += torch.matmul(outputs, features_mean.T)
                predicts = sim
                targets -= self.ncla[t]
            elif t == self.cur_task + 1:
                outputs = F.normalize(outputs_mem, dim=1)
                features_mean = F.normalize(self.model.features_mean_mem, dim=1)
                predicts = torch.matmul(outputs, features_mean.T)
            
        values,indices=predicts.max(1)
        hits=(indices==targets).float()
        return hits.sum().data.cpu().numpy()


    def train_epoch(self, t, data_loader, train_transform, valid_transform, squeeze, lr, mask, mem):
        self.model.train()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = torch.cat([valid_transform(images), train_transform(images)], dim=0)
                targets = torch.cat([targets, targets], dim=0)
            self.train_batch(t, images, targets, squeeze, lr, mask, mem)
        
        if squeeze:
            self.model.squeeze(self.optimizer.state)
            model_count, layers_count = self.model.count_params()


    def eval(self, t, data_loader, valid_transform, mask=True, over_param=True, mem=True):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
                    
            hits = self.eval_batch(t, images, targets, mask, over_param, mem)
            total_acc += hits
            total_num += len(targets)
                
        return 0, total_acc/total_num

    def get_classes_statistic(self, t, data_loader, valid_transform, mask, over_param, mem):
        self.model.eval()
        features = []
        labels = []
        if mask or over_param:
            self.model.get_old_params(t)
        elif mem:
            self.model.get_old_params(self.cur_task+1)
        for images, targets in data_loader:
            images = images.to(device)
            if valid_transform:
                images = valid_transform(images)
            if mask:
                outputs = self.model.forward(images, t, mask=True, mem=False)
            if over_param:
                outputs = self.model.forward(images, t, mask=False, mem=False)
            if mem:
                outputs = self.model.forward(images, self.cur_task+1, mask=True, mem=True)
            features.append(outputs.detach())
            labels.append(targets)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        features_mean = []
        if mask or over_param:
            begin = self.ncla[t]
            end = self.ncla[t+1]
        else:
            begin = 0
            end = self.ncla[-1]
        for cla in range(begin, end):
            ids = (labels == cla)
            cla_features = features[ids]
            features_mean.append(cla_features.mean(0))

        features_mean = torch.stack(features_mean, dim=0).to(device)
        if over_param:
            if self.model.features_mean is None:
                self.model.features_mean = features_mean # [num classes, feature dim]
            else:
                self.model.features_mean = torch.cat([self.model.features_mean[:begin], features_mean], dim=0)
        elif mask:
            if self.model.features_mean_mask is None:
                self.model.features_mean_mask = features_mean # [num classes, feature dim]
            else:
                self.model.features_mean_mask = torch.cat([self.model.features_mean_mask[:begin], features_mean], dim=0)
        elif mem:
            if self.model.features_mean_mem is None:
                self.model.features_mean_mem = features_mean # [num classes, feature dim]
            else:
                self.model.features_mean_mem = torch.cat([self.model.features_mean_mem[:begin], features_mean], dim=0)

    def update_mem(self, data_loader, valid_transform):
        self.model.eval()
        features = []
        labels = []
        samples = []
        for images, targets in data_loader:
            samples.append(images)
            images = images.to(device)
            if valid_transform:
                images = valid_transform(images)
            outputs = self.model.forward(images, self.cur_task, mask=False, mem=False)
            features.append(outputs.detach())
            labels.append(targets)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        samples = torch.cat(samples, dim=0)
        for cla in range(self.ncla[-2], self.ncla[-1]):
            idx = (labels == cla)
            cla_features = features[idx]
            cla_samples = samples[idx]
            cla_labels = labels[idx]
            cl, centroids = KMeans(cla_features, K=self.args.nsamples, Niter=100)
            sim = torch.mm(centroids, cla_features.T)
            v, i  = sim.max(1)
            self.model.mem_images = torch.cat([self.model.mem_images, cla_samples[i.view(-1)]], dim=0)
            self.model.mem_targets = torch.cat([self.model.mem_targets, cla_labels[i.view(-1)]], dim=0)
        print(self.model.mem_targets.shape)

    def sup_con_cl_loss(self, features, labels):
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
    
    def mem_loss(self, features, labels):
        sim = torch.div(
            torch.matmul(features, self.model.features_mean.T),
            self.temperature)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()      
        pos_mask = (labels.view(-1, 1) == torch.arange.view(1, -1)).float().to(device)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

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

