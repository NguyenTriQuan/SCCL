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
import networks.npb_cl_net as network
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
        self.model = Net(input_size=inputsize, taskcla=taskcla, norm_type=args.norm_type).to(device)
        self.inputsize = inputsize
        self.taskcla = taskcla
        
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
        self.sparsity = args.sparsity
        self.alpha = args.alpha 
        self.beta = args.beta

        self.args = args
        self.lamb = float(args.lamb)
        self.check_point = None
        
        self.get_name()
        self.ce = torch.nn.CrossEntropyLoss()

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
            self.model.prune(self.inputsize, self.sparsity, self.alpha, self.beta, node_constraint=False, 
                max_param_per_kernel=None, min_param_to_node=None, is_store_mask=False, file_name=None)
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
        else: 
            print('Continue training current task')

        self.model = self.model.to(device)
        self.ncla = self.model.ncla
        self.cur_task = len(self.ncla)-2
        self.threshold = np.array([0.99] * 6) + t*np.array([0.003] * 6)
        print(self.log_name)

        self.mean = train_loader.dataset.tensors[0].mean(dim=(0, 2, 3))
        var = train_loader.dataset.tensors[0].var(dim=(0, 2, 3))
        next_ks = self.model.DM[0].ks
        self.std = (var.sum() * next_ks) ** 0.5
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform)
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
        outputs = self.model.forward(images, t)
        loss = self.ce(outputs, targets) 
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
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
                    
            hits = self.eval_batch(t, images, targets)
            total_acc += hits
            total_num += len(targets)
                
        return 0, total_acc/total_num
