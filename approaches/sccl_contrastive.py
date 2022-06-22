import sys, time, os
import math

import numpy as np
from pytest import param
from sympy import arg
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn

import time
import csv
from utils import *
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
import networks.sccl_contrastive_net as network
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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

        self.args = args
        self.lambs = [float(i) for i in args.lamb.split('_')]
        self.check_point = None
        
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.shape_out = self.model.layers[-1].shape_out
        self.cur_task = len(self.shape_out)-1

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

    def train(self, t, train_loader, valid_loader, ncla=0):

        if self.check_point is None:
            print('Training new task')

            self.model.expand(ncla)

            self.shape_out = self.model.layers[-1].shape_out
            self.cur_task = len(self.shape_out)-1

            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}

            try:
                os.remove(f'../result_data/trained_model/{self.log_name}.model')
            except:
                pass
            self.get_name(t)
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
                
            with open(f'../result_data/csv_data/{self.log_name}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['train loss', 'train acc', 'valid loss', 'valid acc', 'fro norm'])
        else: 
            print('Retraining current task')

        self.model = self.model.to(device)
        self.model = accelerator.prepare(self.model)
        self.model.restrict_gradients(t-1, False)
        self.shape_out = self.model.layers[-1].shape_out
        self.cur_task = len(self.shape_out)-1

        self.lamb = self.lambs[self.cur_task-1]
        print('lambda', self.lamb)
        print(self.log_name)

        train_loader = accelerator.prepare(train_loader)
        valid_loader = accelerator.prepare(valid_loader)

        self.train_phase(t, train_loader, valid_loader, True)
        if not self.check_point['squeeze']:
            self.check_point = None
            return 

        self.prune(t, train_loader, thres=self.thres)


        self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.train_phase(t, train_loader, valid_loader, False)

        self.check_point = None
        

    def train_phase(self, t, train_loader, valid_loader, squeeze):

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)

        train_loss,train_acc=self.eval(t,train_loader)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t,valid_loader)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        # self.optimizer = self.check_point['optimizer']
        self.optimizer = self._get_optimizer(lr)
        start_epoch = self.check_point['epoch'] + 1
        squeeze = self.check_point['squeeze']

        if squeeze:
            best_acc = train_acc
        else:
            best_acc = valid_acc
    
        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(t, train_loader, squeeze)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(t, train_loader)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                
                s_H = self.model.s_H()
                print('s_H={:.1e}'.format(s_H), end='')
                # Adapt lr
                if squeeze:
                    if train_acc >= best_acc:
                        best_acc = train_acc
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
                with open(f'../result_data/csv_data/{self.log_name}.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([train_loss, train_acc, valid_loss, valid_acc, s_H])

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']

    def train_batch(self, t, images, targets, squeeze):
        outputs = self.model.forward(images, t=t)
        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

        loss = self.ce(outputs, targets)

        if squeeze:
            loss += self.model.group_lasso_reg() * self.lamb
                
        self.optimizer.zero_grad()
        loss.backward() 
        accelerator.backward(loss)
        self.optimizer.step()

    def eval_batch(self, t, images, targets):
        if t is None:
            outputs = []
            entropy = []
            aug_images = [images]
            batch_DA = 32
            for n in range(batch_DA):
                aug_images.append(self.trans(images))
            aug_images = torch.cat(aug_images, dim=0)
            for task in range(1, self.cur_task + 1):
                self.model.get_params(task-1)
                output = self.model.forward(aug_images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                output = output.reshape(batch_DA+1, len(targets), -1)
                # output = F.softmax(output, dim=-1)
                output = F.softplus(output)
                output = output / output.sum(-1).unsqueeze(-1)
                outputs.append(output[0])
                entropy.append(-(output*output.log()).sum((-1, 0)))

            entropy = torch.stack(entropy, dim=1)
            outputs = torch.stack(outputs, dim=1)
            v, i = entropy.min(1)
            outputs = outputs[range(outputs.shape[0]), i]
        else:
            self.model.get_params(t-1)
            outputs = self.model.forward(images, t=t)
            outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
                        
        loss=self.ce(outputs,targets)
        values,indices=outputs.max(1)
        hits=(indices==targets).float()

        return loss.data.cpu().numpy()*len(targets), hits.sum().data.cpu().numpy()

    def train_epoch(self, t, data_loader, squeeze=True):
        self.model.train()
        self.model.get_params(t-1)
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            self.train_batch(t, images, targets, squeeze)


    def eval(self, t, data_loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
                    
            loss, hits = self.eval_batch(t, images, targets)
            total_loss += loss
            total_acc += hits
            total_num += len(targets)
                
        return total_loss/total_num,total_acc/total_num


    def prune(self, t, data_loader, thres=0.0):

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


