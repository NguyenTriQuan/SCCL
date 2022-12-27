from cProfile import label
import sys, time, os
import math
from turtle import pos

import numpy as np
from pytest import param
from sympy import arg
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import kornia as K

import time
import csv
from utils import *
from sccl_con_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
import networks.sccl_con_net as network
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.distributions import Normal

from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device

import sys

class Appr(object):

    def __init__(self,inputsize=None,taskcla=None,args=None,thres=1e-3,lamb='0',nepochs=100,sbatch=256,val_sbatch=256,
                lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=10,optim='Adam',tasknum=1,fix=False,norm_type=None):

        Net = getattr(network, args.arch)
        self.model = Net(input_size=inputsize, norm_type=args.norm_type).to(device)

        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
        self.lr_min = lr/100
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience 
        self.clipgrad = clipgrad
        self.optim = args.optimizer
        self.thres = args.thres
        self.tasknum = args.tasknum
        self.fix = args.fix
        self.experiment = args.experiment
        self.approach = args.approach
        self.arch = args.arch
        self.seed = args.seed
        self.norm_type = args.norm_type
        self.temperature = 0.15
        self.contrast_mode = 'all'
        self.base_temperature = 0.07

        self.args = args
        self.lambs = [float(i) for i in args.lamb.split('_')]
        self.check_point = None
        self.ncla = self.model.ncla
        
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.get_name(self.tasknum+1)

    def get_name(self, t):
        self.log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_optim_{}_fix_{}_norm_{}'.format(self.experiment, self.approach, self.arch, self.seed,
                                                                                '_'.join([str(lamb) for lamb in self.lambs[:t]]),  
                                                                                self.lr, self.batch_size, self.nepochs, self.optim, self.fix, self.norm_type)
        
    def resume(self):
        for t in range(1, self.tasknum + 1):
            try:
                self.get_name(t)

                self.check_point = torch.load(f'../result_data/trained_model/{self.log_name}.model')
                self.model = self.check_point['model']
                print('Resume from task', t-1)

                return t-1
            except:
                continue
        return 0

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr

        params = self.model.get_optim_params()

        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)

        optimizer = accelerator.prepare(optimizer)
        return optimizer

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

        if self.check_point is None:
            print('Training new task')

            self.model.expand(ncla)
            self.model = self.model.to(device)
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}

            try:
                os.remove(f'../result_data/trained_model/{self.log_name}.model')
            except:
                pass
            self.get_name(t)
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
                
        else: 
            print('Retraining current task')

        self.ncla = self.model.ncla
        self.n_old = self.model.ncla[t-1]
        print(self.n_old)
        self.model = accelerator.prepare(self.model)
        self.model.restrict_gradients(t-1, False)
        self.lamb = self.lambs[t-1]
        print('lambda', self.lamb)
        print(self.log_name)
        # print(train_transform)
        # print(valid_transform)

        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)

        # self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, False)
        # if not self.check_point['squeeze']:
        #     self.check_point = None
        #     return 

        # self.prune(t, train_loader, valid_transform, thres=self.thres)


        # self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        # torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, False)

        self.check_point = None
        

    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform, squeeze):

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)

        self.get_classes_statistic(t, train_loader, valid_transform)
        _, train_acc=self.eval(t, train_loader, valid_transform)
        _, valid_acc=self.eval(t, valid_loader, valid_transform)

        print('Train: acc={:5.2f}% | Valid: acc={:5.2f}% |'.format(100*train_acc, 100*valid_acc))
        best_acc = valid_acc

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        # self.optimizer = self.check_point['optimizer']
        self.optimizer = self._get_optimizer(lr)
        start_epoch = self.check_point['epoch'] + 1
        # squeeze = self.check_point['squeeze']        
        squeeze = False    

        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                loss = self.train_epoch(t, train_loader, train_transform, squeeze)
                clock1=time.time()
                self.get_classes_statistic(t, train_loader, valid_transform)
                _, train_acc=self.eval(t, train_loader, valid_transform)
                _, valid_acc=self.eval(t, valid_loader, valid_transform)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),loss, 100*train_acc),end='')
                print(' Valid: acc={:5.2f}% |'.format(100*valid_acc),end='')
                # s_H = self.model.s_H()
                # print('s_H={:.1e}'.format(s_H), end='')
                # Adapt lr
                if squeeze:
                    if valid_acc >= best_acc:
                        best_acc = valid_acc
                        self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                        print(' *', end='')
                        patience = self.lr_patience

                else:
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                        patience = self.lr_patience
                        print(' *', end='')
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

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        self.get_classes_statistic(t, train_loader, valid_transform)

    def get_classes_statistic(self, t, data_loader, valid_transform):
        self.model.eval()
        self.model.get_params(t-1)
        repres = []
        labels = []
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
            features = self.model.forward(images, t=t)
            repres.append(features.detach())
            labels.append(targets)

        repres = torch.cat(repres, dim=0)
        labels = torch.cat(labels, dim=0)
        repres_mean = []
        repres_std = []
        for c in range(self.ncla[t-1], self.ncla[t]):
            ids = (labels == c)
            repres_c = repres[ids]
            repres_mean.append(repres_c.mean(0))
            repres_std.append(repres_c.std(0))

        repres_mean = torch.stack(repres_mean, dim=0)
        repres_std = torch.stack(repres_std, dim=0)
        if self.model.repres_mean is None:
            self.model.repres_mean = repres_mean
            self.model.repres_std = repres_std
        else:
            self.model.repres_mean = torch.cat([self.model.repres_mean[:self.ncla[t-1]], repres_mean], dim=0)
            self.model.repres_std = torch.cat([self.model.repres_std[:self.ncla[t-1]], repres_std], dim=0)

        # print(self.model.repres_mean.shape)

    def train_batch(self, t, images, targets, squeeze):
        features = self.model.forward(images, t=t)
        batch_size = targets.shape[0]
        # f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # loss = self.SupConLoss(features, targets)

        if self.n_old != 0 :
            old_feat_dist = Normal(self.model.repres_mean[:self.ncla[t-1]], self.model.repres_std[:self.ncla[t-1]])
            old_features = old_feat_dist.sample(torch.Size([3 * (len(self.ncla)-1) * batch_size // self.n_old])).view(-1, features.shape[1]).to(device)
            old_targets = torch.arange(self.n_old).repeat(3 * (len(self.ncla)-1) * batch_size // self.n_old).view(-1).to(device)
            features = torch.cat([features, old_features], dim=0)
            targets = torch.cat([targets, old_targets], dim=0)

        features = F.normalize(features, dim=1)
        loss = self.sup_con_cl_loss(features, targets)
        # if squeeze:
        #     loss += self.model.group_lasso_reg() * self.lamb
                
        self.optimizer.zero_grad()
        # loss.backward() 
        accelerator.backward(loss)
        self.optimizer.step()
        return loss.data.cpu().numpy()*batch_size

    def eval_batch(self, t, images, targets):
        if t is not None:
            self.model.get_params(t-1)
            features = self.model.forward(images, t=t)

            features = F.normalize(features, dim=1)
            feature_mean = F.normalize(self.model.repres_mean, dim=1)
            sim = torch.matmul(features, feature_mean.T)

            # feat_dist = Normal(self.model.repres_mean, self.model.repres_std)
            # features = features.unsqueeze(1).expand([features.shape[0], self.ncla[-1], features.shape[1]])
            # log_prob = feat_dist.log_prob(features)
            # sim = log_prob.sum(2)
            v, i = sim.max(1)

            # feat_dist = Normal(self.model.repres_mean[self.ncla[t-1]: self.ncla[t]], self.model.repres_std[self.ncla[t-1]: self.ncla[t]])
            # features = features.unsqueeze(1).expand([features.shape[0], self.ncla[t]-self.ncla[t-1], features.shape[1]])
            # log_prob = feat_dist.log_prob(features)
            # sim = log_prob.sum(2)
            # v, i = sim.max(1) 
            # i += self.ncla[t-1]

        else:
            sim = []
            entropy = []
            for t in range(1, len(self.ncla)):
                self.model.get_params(t-1)
                features = self.model.forward(images, t=t)

                features = F.normalize(features, dim=1)
                feature_mean = F.normalize(self.model.repres_mean, dim=1)
                prob = torch.matmul(features, feature_mean.T)
                sim.append(prob)
                prob = F.softmax(prob*2, dim=1)
                entropy.append((-prob*prob.log()).sum(1))

                # feat_dist = Normal(self.model.repres_mean, self.model.repres_std)
                # features = features.unsqueeze(1).expand([features.shape[0], self.ncla[-1], features.shape[1]])
                # log_prob = feat_dist.log_prob(features).sum(2)
                # sim.append(log_prob)
                # log_prob = log_prob/sum(log_prob)
                # log_prob = F.softmax(log_prob*10000, dim=1)
                # print(log_prob)
                # entropy.append((-log_prob*log_prob.log()).sum(1))

            sim = torch.stack(sim, dim=1)
            entropy = torch.stack(entropy, dim=1)
            v, i = entropy.min(1)
            sim = sim[range(sim.shape[0]), i]
            v, i = sim.max(1)
        
        hits = (i==targets).float()
        return hits.sum().data.cpu().numpy()

    def train_epoch(self, t, data_loader, train_transform, valid_transform, squeeze=True):
        self.model.train()
        self.model.get_params(t-1)
        total_loss=0
        total_num=0
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            if train_transform:
                images = torch.cat([valid_transform(images), train_transform(images)], dim=0)
                targets = torch.cat([targets, targets], dim=0)
                # images = torch.cat([images, images], dim=0)
                # images = train_transform(images)
            total_loss += self.train_batch(t, images, targets, squeeze)
            total_num += targets.shape[0]
        return total_loss/total_num


    def eval(self, t, data_loader, valid_transform):
        total_acc=0
        total_num=0
        self.model.eval()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
            hits = self.eval_batch(t, images, targets)
            total_acc += hits
            total_num += len(targets)
                
        return 0, total_acc/total_num
        
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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


    def prune(self, t, data_loader, valid_transform, thres=0.0):

        fig, axs = plt.subplots(3, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 9))
        for i, m in enumerate(self.model.DM[:-1]):
            axs[0][i].hist(m.norm_in().detach().cpu().numpy(), bins=100)
            axs[0][i].set_title(f'layer {i+1}')

            axs[1][i].hist(m.norm_out().detach().cpu().numpy(), bins=100)
            axs[1][i].set_title(f'layer {i+1}')

            axs[2][i].hist((m.norm_in()*m.norm_out()).detach().cpu().numpy(), bins=100)
            axs[2][i].set_title(f'layer {i+1}')

        plt.show()

        loss,acc=self.eval(t,data_loader)
        loss, acc = round(loss, 3), round(acc, 3)
        print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
        # pre_prune_acc = acc
        pre_prune_loss = loss
        prune_ratio = np.ones(len(self.model.DM)-1)
        step = 0
        pre_sum = 0
        # Dynamic expansion
        while True:
            t1 = time.time()
            fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 2))
            print('Pruning ratio:', end=' ')
            for i in range(0, len(self.model.DM)-1):
                m = self.model.DM[i]
                mask_temp = m.mask
                norm = m.get_importance()

                low = 0 
                if m.mask is None:
                    high = norm.shape[0]
                else:
                    high = int(sum(m.mask))

                axs[i].hist(norm.detach().cpu().numpy(), bins=100)
                axs[i].set_title(f'layer {i+1}')

                if norm.shape[0] != 0:
                    values, indices = norm.sort(descending=True)
                    loss,acc=self.eval(t,data_loader)
                    loss, acc = round(loss, 3), round(acc, 3)
                    pre_prune_loss = loss

                    while True:
                        k = (high+low)//2
                        # Select top-k biggest norm
                        m.mask = (norm>values[k])
                        loss, acc = self.eval(t, data_loader)
                        loss, acc = round(loss, 3), round(acc, 3)
                        # post_prune_acc = acc
                        post_prune_loss = loss
                        if  post_prune_loss <= pre_prune_loss:
                        # if pre_prune_acc <= post_prune_acc:
                            # k is satisfy, try smaller k
                            high = k
                            # pre_prune_loss = post_prune_loss
                        else:
                            # k is not satisfy, try bigger k
                            low = k

                        if k == (high+low)//2:
                            break


                if high == norm.shape[0]:
                    # not found any k satisfy, keep all neurons
                    m.mask = mask_temp
                else:
                    # found k = high is the smallest k satisfy
                    m.mask = (norm>values[high])

                # remove neurons 
                # m.squeeze()

                if m.mask is None:
                    prune_ratio[i] = 0.0
                else:
                    mask_count = int(sum(m.mask))
                    total_count = m.mask.numel()
                    prune_ratio[i] = 1.0 - mask_count/total_count

                print('{:.3f}'.format(prune_ratio[i]), end=' ')
                # m.mask = None

            fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}.pdf', bbox_inches='tight')
            # plt.show()
            loss,acc=self.eval(t,data_loader)
            print('| Post Prune: loss={:.3f}, acc={:5.2f}% | Time={:5.1f}ms |'.format(loss, 100*acc, (time.time()-t1)*1000))

            step += 1
            if sum(prune_ratio) == pre_sum:
                break
            pre_sum = sum(prune_ratio)

        for m in self.model.DM[:-1]:
            m.squeeze()
            m.mask = None
        loss,acc=self.eval(t,data_loader)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)



