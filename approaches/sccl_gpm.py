from cProfile import label
import sys, time, os
import math

import numpy as np
from pytest import param
from sqlalchemy import false
from sympy import arg
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import kornia as K

import time
import csv
from utils import *
from layers.sccl_gpm_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
import networks.sccl_gpm_net as network
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from gmm_torch.gmm import GaussianMixture
# from pykeops.torch import LazyTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.tasknum = args.tasknum
        self.fix = args.fix
        self.experiment = args.experiment
        self.approach = args.approach
        self.arch = args.arch
        self.seed = args.seed
        self.norm_type = args.norm_type
        self.ablation = args.ablation

        self.args = args
        self.lambs = [float(i) for i in args.lamb.split('_')]
        self.thresholds = [float(i) for i in args.threshold.split('_')]
        self.threshold = args.threshold
        self.check_point = None
        
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.shape_out = self.model.DM[-1].shape_out
        self.cur_task = len(self.shape_out)-1

        self.get_name(self.tasknum+1)

    def get_name(self, t):
        self.log_name = '{}_{}_{}_{}_{}_lamb_{}_thres_{}_lr_{}_batch_{}_epoch_{}_optim_{}_fix_{}_norm_{}'.format(
                                        self.experiment, self.approach, self.ablation, self.arch, self.seed,
                                                '_'.join([str(lamb) for lamb in self.lambs[:t]]), self.threshold, 
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

        params = self.model.get_optim_params(self.ablation)
        # scales = [m.scale[-1] for m in self.model.DM]
        # optim_params = [{'params':params, 'lr':lr}, {'params':scales, 'lr':0.1}]
        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr,
                          weight_decay=0.0, momentum=0.9)
            # optimizer = torch.optim.SGD(params, lr=lr)
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)
            # optimizer = Adam_(self.model, lr=lr)

        return optimizer

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

        if self.check_point is None:
            print('Training new task')
            self.model.expand(ncla, self.ablation)
            self.model = self.model.to(device)
            self.shape_out = self.model.DM[-1].shape_out
            self.cur_task = len(self.shape_out)-1

            # self.get_sim(train_loader, valid_transform, t)

            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}

            try:
                os.remove(f'../result_data/trained_model/{self.log_name}.model')
            except:
                pass
            self.get_name(t)
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
                
        else: 
            print('Retraining current task')

        print(self.model.report())
        self.model = self.model.to(device)
        self.shape_out = self.model.DM[-1].shape_out
        self.cur_task = len(self.shape_out)-1

        self.lamb = self.lambs[self.cur_task-1]
        print('lambda', self.lamb)
        print(self.log_name)
        self.model.squeeze(self.optimizer.state)

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, True)
        if not self.check_point['squeeze']:
            self.check_point = None
            return 

        self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, False)

        if 'gpm' not in self.ablation:
            self.updateGPM(train_loader, valid_transform, self.thresholds)
            self.check_point['model'] = self.model
            torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.check_point = None
        for m in self.model.DM:
            print(m.fwt_weight[-1].norm(2))        

    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform, squeeze):

        self.model.count_params()
        self.model.count_GPM()

        train_loss,train_acc=self.eval(t,train_loader,valid_transform)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t,valid_loader,valid_transform)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        self.optimizer = self.check_point['optimizer']
        start_epoch = self.check_point['epoch'] + 1
        squeeze = self.check_point['squeeze']

        if squeeze:
            best_acc = train_acc
        else:
            best_acc = valid_acc
    
        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(t, train_loader, train_transform, squeeze, lr)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(t, train_loader, valid_transform)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if squeeze:
                    self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                    torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                    model_count, layers_count = self.model.count_params()
                    self.logger.log_metric('num params', model_count, epoch=e)
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
                self.logger.log_metrics({
                    'train acc':train_acc,
                    'valid acc':valid_acc
                }, epoch=e)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']

    def train_batch(self, t, images, targets, squeeze, lr):
        outputs = self.model.forward(images, t=t)
        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
        if self.args.cil:
            targets -= sum(self.shape_out[:t])

        loss = self.ce(outputs, targets)

        # if squeeze:
        #     loss += self.model.group_lasso_reg() * self.lamb
                
        self.optimizer.zero_grad()
        loss.backward() 
        self.model.project_gradient()
        self.optimizer.step()
        if squeeze:
            self.model.proximal_gradient_descent(lr, self.lamb)
        
        for i, m in enumerate(self.model.DM):
            if m.projection_matrix is not None:
                self.logger.log_metric(f'layer {i} sim', m.cos_sim)
            

    def eval_batch(self, t, images, targets):
        if t is None:
            outputs = self.model.forward(images, t=self.cur_task)
        else:
            outputs = self.model.forward(images, t=t)
            outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
            if self.args.cil:
                targets -= sum(self.shape_out[:t])

        loss=self.ce(outputs,targets)
        values,indices=outputs.max(1)
        hits=(indices==targets).float()

        return loss.data.cpu().numpy()*len(targets), hits.sum().data.cpu().numpy()

    def train_epoch(self, t, data_loader, train_transform, squeeze, lr):
        self.model.train()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = train_transform(images)
            self.train_batch(t, images, targets, squeeze, lr)
        
        if squeeze:
            self.model.squeeze(self.optimizer.state)


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
                    
            loss, hits = self.eval_batch(t, images, targets)
            total_loss += loss
            total_acc += hits
            total_num += len(targets)
                
        return total_loss/total_num,total_acc/total_num

    def test(self, data_loader, valid_transform):
        self.model.eval()

        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)

            outputs = self.model.forward(images, t=self.cur_task)
            # for m in self.model.DM:
            #     print(m.act)
            print(outputs)
            break

    def updateGPM (self, data_loader, valid_transform, thresholds): 
        # Collect activations by forward pass
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
        outputs  = self.model(inputs, t=self.cur_task)

        self.model.get_feature(thresholds)
        self.model.track_input(False)
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i, m in enumerate(self.model.DM):
            if m.feature is not None:
                print ('Layer {} : {}/{}'.format(i+1, m.feature.shape[1], m.feature.shape[0]))
        print('-'*40)

    def get_sim(self, data_loader, valid_transform, t): 
        # Collect activations by forward pass
        inputs = []
        labels = []
        N = 0
        for i, (images, targets) in enumerate(data_loader):
            inputs.append(images)
            labels.append(targets)
            N += images.shape[0]
            if N >= self.val_batch_size:
                break
        self.model.eval()
        inputs = torch.cat(inputs, dim=0).to(device)[:self.val_batch_size]
        labels = torch.cat(labels, dim=0).to(device)[:self.val_batch_size]
        if valid_transform:
            inputs = valid_transform(inputs)
        outputs = self.model.forward(inputs, t=t)
        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
        if self.args.cil:
            labels -= sum(self.shape_out[:t])

        loss = self.ce(outputs, labels)

        # if squeeze:
        #     loss += self.model.group_lasso_reg() * self.lamb
                
        self.optimizer.zero_grad()
        loss.backward() 
        self.model.compute_project_similarity(t)



    def prune(self, t, data_loader, valid_transform, thres=0.0):

        loss,acc=self.eval(t,data_loader,valid_transform)
        loss, acc = round(loss, 3), round(acc, 3)
        print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
        # pre_prune_acc = acc
        pre_prune_loss = loss
        prune_ratio = np.ones(len(self.model.DM)-1)
        step = 0
        pre_sum = 0
        for i in range(0, len(self.model.DM)-1):
            m = self.model.DM[i]
            m.mask = torch.ones(m.shape_out[-1]-m.shape_out[-2]).bool().cuda()
        while True:
            t1 = time.time()
            # fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 2))
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

                # axs[i].hist(norm[m.mask].detach().cpu().numpy(), bins=100)
                # axs[i].set_title(f'layer {i+1}')
                if norm.shape[0] != 0:
                    values, indices = norm.sort(descending=True)
                    loss,acc=self.eval(t,data_loader,valid_transform)
                    loss, acc = round(loss, 3), round(acc, 3)
                    pre_prune_loss = loss

                    while True:
                        k = (high+low)//2
                        # Select top-k biggest norm
                        m.mask = (norm>values[k])
                        loss, acc = self.eval(t, data_loader, valid_transform)
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

            # fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}.pdf', bbox_inches='tight')
            # plt.show()
            loss,acc=self.eval(t,data_loader,valid_transform)
            print('| Post Prune: loss={:.3f}, acc={:5.2f}% | Time={:5.1f}ms |'.format(loss, 100*acc, (time.time()-t1)*1000))

            step += 1
            if sum(prune_ratio) == pre_sum:
                break
            pre_sum = sum(prune_ratio)

        self.model.squeeze()

        loss,acc=self.eval(t,data_loader,valid_transform)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.count_params()
        print('num params', params)


    


