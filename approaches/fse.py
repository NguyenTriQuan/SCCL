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
import networks.fse_net as network
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
        
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
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
        if self.lr_rho > 0:
            params += self.model.get_optim_scales(lr*self.lr_rho)
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
            if t > 0:
                self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
                self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=False, mask=True)
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

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=True, mask=False)
        if not self.check_point['squeeze']:
            self.check_point = None
            return 

        self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
        if 'phase2' not in self.ablation:
            self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, squeeze=False, mask=False)

        self.model.freeze(t)
        self.check_point = None  

        self.model.count_params()
        self.model.get_old_params(t)
        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask=True, mask_only=False)
        print(' Valid ensemble: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))
        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask=False, mask_only=False)
        print(' Valid no ensemble: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))
        valid_loss,valid_acc=self.eval(None, valid_loader, valid_transform, mask=True, mask_only=False)
        print(' Valid ensemble no task identity: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))
        valid_loss,valid_acc=self.eval(None, valid_loader, valid_transform, mask=False, mask_only=False)
        print(' Valid no ensemble no task identity: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        self.model.update_scale()
        for n, m in enumerate(self.model.DM[:-1]):
            print('layer', n)
            for i in range(m.cur_task+1):
                for j in range(m.cur_task+1):
                    print(m.scale[i][j], end=' ')
                print()
            print()


    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform, squeeze, mask):
        if mask:
            mask_only = True
        else:
            mask_only = False
        self.model.count_params()
        self.model.get_old_params(t)
        train_loss,train_acc=self.eval(t, train_loader, valid_transform, mask, mask_only)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask, mask_only)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        # if mask:
        #     self.nepochs = 50
        # else:
        #     self.nepochs = self.args.nepochs

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

        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(t, train_loader, train_transform, squeeze, lr, mask)
                clock1=time.time()
                train_loss,train_acc=self.eval(t, train_loader, valid_transform, mask, mask_only)
                clock2=time.time()
                print('| Epoch {:2d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform, mask, mask_only)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if squeeze:
                    self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                    torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                    # model_count, layers_count = self.model.count_params()
                    # if self.logger is not None:
                    #     self.logger.log_metric('num params', model_count, epoch=e)
                elif mask:
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

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        print(train_accs)
        print(valid_accs)

    def train_batch(self, t, images, targets, squeeze, lr, mask):
        outputs = self.model.forward(images, t, mask)
        loss = self.ce(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
        if squeeze:
            self.model.proximal_gradient_descent(lr, self.lamb)

    def eval_batch(self, t, images, targets, mask=True, mask_only = False):
        self.model.eval()
        with torch.no_grad():
            if t is None:
                joint_entropy_tasks = []
                outputs_tasks = []
                for i in range(self.cur_task+1):
                    self.model.get_old_params(i)
                    outputs = []
                    if mask and i > 0:
                        outputs += [self.model.forward(images, i, mask=True)]
                    if not mask_only:
                        outputs += [self.model.forward(images, i, mask=False)]
                    outputs = ensemble_outputs(torch.stack(outputs, dim=-1))
                    outputs_tasks += [outputs]
                    outputs = torch.exp(outputs)
                    joint_entropy = -torch.sum(outputs * torch.log(outputs+0.0001), dim=1)
                    if i == 0 and mask:
                        joint_entropy *= self.args.factor
                    joint_entropy_tasks.append(joint_entropy)

                outputs_tasks = torch.stack(outputs_tasks, dim=1)
                joint_entropy_tasks = torch.stack(joint_entropy_tasks)
                joint_entropy_tasks = joint_entropy_tasks.transpose(0, 1)
                predicted_task = torch.argmin(joint_entropy_tasks, axis=1)
                outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            else:
                outputs = []
                if mask and t > 0:
                    outputs += [self.model.forward(images, t, mask=True)]
                if not mask_only:
                    outputs += [self.model.forward(images, t, mask=False)]

                outputs = ensemble_outputs(torch.stack(outputs, dim=-1))
        loss=self.ce(outputs,targets)
        values,indices=outputs.max(1)
        hits=(indices==targets).float()
        return loss.data.cpu().numpy()*len(targets), hits.sum().data.cpu().numpy()


    def train_epoch(self, t, data_loader, train_transform, squeeze, lr, mask):
        self.model.train()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = train_transform(images)
            self.train_batch(t, images, targets, squeeze, lr, mask)
        
        if squeeze:
            self.model.squeeze(self.optimizer.state)
            model_count, layers_count = self.model.count_params()


    def eval(self, t, data_loader, valid_transform, mask=True, mask_only=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
                    
            loss, hits = self.eval_batch(t, images, targets, mask, mask_only)
            total_loss += loss
            total_acc += hits
            total_num += len(targets)
                
        return total_loss/total_num,total_acc/total_num

