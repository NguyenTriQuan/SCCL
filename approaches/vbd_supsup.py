import sys, time, os
import math

import numpy as np
from pytest import param
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
import time
import csv
from utils import *
from networks.vbd_supsup_net import VBD_Layer
import networks.vbd_supsup_net as network
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
# import pygame
# from visualize import draw

import sys
from arguments import get_args
args = get_args()
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

class Appr(object):

    def __init__(self,inputsize=None,taskcla=None,args=None,thres=1e-3,lamb='0',nepochs=100,sbatch=256,val_sbatch=256,
                lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=10,optim='Adam',tasknum=1,fix=False):

        Net = getattr(network, args.arch)
        self.model = Net(inputsize, taskcla).to(device)

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
        self.experiment = args.experiment
        self.approach = args.approach
        self.arch = args.arch
        self.seed = args.seed

        self.args = args
        self.lambs = [float(i) for i in args.lamb.split('_')]
        self.check_point = None
        
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(-1)

        
    def resume(self):
        for t in range(0, self.tasknum + 1):
            try:
                self.log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_optim_{}'.format(self.experiment, self.approach, self.arch, self.seed,
                                                                                '_'.join([str(lamb) for lamb in self.lambs[:t]]),  
                                                                                self.lr, self.batch_size, self.nepochs, self.optim)

                self.check_point = torch.load(f'../result_data/trained_model/{self.log_name}.model')
                self.model = self.check_point['model']
                print('Resume from task', t)

                return t
            except:
                continue
        return 0
        
    def _get_optimizer(self,t,lr=None):
        if lr is None: lr=self.lr
        # params = []
        # for m in self.model.modules():
        #     if isinstance(m, VBD_Layer):
        #         params += [m.log_sigma2, m.mu]
                
        # params += [self.model.last[t].weight, self.model.last[t].bias]
        params = self.model.parameters()

        if self.optim == 'SGD':
            return torch.optim.SGD(params, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)
        if self.optim == 'Adam':
            return torch.optim.Adam(params, lr=lr)

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

        print(self.check_point)
        if self.check_point is None:
            print('Training new task')

            self.check_point = {'model':self.model, 'optimizer':self._get_optimizer(t), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}

            try:
                os.remove(f'../result_data/trained_model/{self.log_name}.model')
            except:
                pass
            self.log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_optim_{}'.format(self.experiment, self.approach, self.arch, self.seed,
                                                                                self.lambs[0],  
                                                                                self.lr, self.batch_size, self.nepochs, self.optim)
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
                
            with open(f'../result_data/csv_data/{self.log_name}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['train loss', 'train acc', 'valid loss', 'valid acc', 'fro norm'])
        else: 
            print('Retraining current task')


        self.lamb = self.lambs[t]
        print('lambda', self.lamb)
        print(self.log_name)

        train_loss,train_acc=self.eval(t, train_loader, valid_transform)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        self.optimizer = self.check_point['optimizer']
        start_epoch = self.check_point['epoch'] + 1
        # scheduler = CosineAnnealingLR(self.optimizer, T_max=self.nepochs)
        best_acc = valid_acc
    
        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(t, train_loader, train_transform)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(t, train_loader, valid_transform)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                
                # Adapt lr
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'epoch':e, 'lr':lr, 'patience':patience}
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
                        self.optimizer = self._get_optimizer(t, lr)

                print()
                # scheduler.step()
                with open(f'../result_data/csv_data/{self.log_name}.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([train_loss, train_acc, valid_loss, valid_acc])

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']
            self.model.to(device)

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        self.check_point = None

    def train_batch(self, t, images, targets):
        outputs = self.model.forward(images, t=t)

        loss = self.ce(outputs, targets)

        loss += self.kl_divergence() * self.lamb
                
        self.optimizer.zero_grad()
        loss.backward() 

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
                output = self.model.forward(aug_images, t=task)
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
            outputs = self.model.forward(images, t=t)
                        
        loss=self.ce(outputs,targets)
        values,indices=outputs.max(1)
        hits=(indices==targets).float()

        return loss.data.cpu().numpy()*len(targets), hits.sum().data.cpu().numpy()

    def train_epoch(self, t, data_loader, train_transform):
        self.model.train()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            images = train_transform(images)
            self.train_batch(t, images, targets)

        for m in self.model.layers:
            if isinstance(m, VBD_Layer):
                print(m.get_mask().sum().item(), end=' ')
        print()


    def eval(self, t, data_loader, valid_transform):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            images = valid_transform(images)
                    
            loss, hits = self.eval_batch(t, images, targets)
            total_loss += loss
            total_acc += hits
            total_num += len(targets)
                
        return total_loss/total_num,total_acc/total_num

    def kl_divergence(self):
        reg = 0
        strength = 0
        for m in self.model.modules():
            if isinstance(m, VBD_Layer):
                reg += m.kl_divergence() * (m.in_features ** 2)
                strength += m.in_features ** 2
        return reg / strength