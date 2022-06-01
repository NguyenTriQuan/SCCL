import sys, time, os
import math

import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
import time
import csv
from utils import cross_entropy, naive_lip
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer, DynamicBatchNorm
import matplotlib.pyplot as plt
# import pygame
# from visualize import draw


class Appr(object):

    def __init__(self,model,args=None,thres=1e-3,lamb=0,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=10,optim='Adam'):
        self.model=model.to(device)

        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = lr/100
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience 
        self.clipgrad = clipgrad
        self.optim = args.optimizer
        self.thres = args.thres
        self.args = args
        
        self.lambs = [float(i) for i in args.lamb.split('_')]
        if len(self.lambs) < args.tasknum:
            self.lambs = [self.lambs[-1] if i>=len(self.lambs) else self.lambs[i] for i in range(args.tasknum)]

        print('lambs:', self.lambs)

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.max_params = 0
        if args.max_params > 0:
            max_bound = list(self.model.bound) + [sum(past_ncla)]
            add_in = 0
            for i, m in enumerate(self.model.DM):
                add_out = int(max_bound[i])
                self.max_params += m.num_add(add_in, add_out)
                if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                    add_in = self.model.smid * self.model.smid * add_out
                else:
                    add_in = add_out

            # print('SCCL with limited training parameters')
            # print('max_bound', max_bound)
            self.max_params = self.max_params*args.max_params
            print('Max params', self.max_params)
        

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr,
                          weight_decay=0.0, momentum=0.9)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t=None, xtrain=None, ytrain=None, xvalid=None, yvalid=None, train_loader=None, valid_loader=None, prune_loader=None, ncla=0):

        if self.check_point['phase'] == 3:
            print('Training new task')

            self.model.new_task()
            self.model.expand(ncla, self.args.max_mul, self.max_params)
            self.model.to(device)

            self.shape_out = self.model.layers[-1].shape_out
            self.cur_task = len(self.shape_out)-1

            if t > 1:
                # self.get_grad(t, xtrain, ytrain)
                # self.nepochs = 1
                # self.train_phase(t, xtrain, ytrain, xvalid, yvalid, train_loader, valid_loader, prune_loader, True)
                # self.nepochs = self.args.nepochs
                self.prune_previous(t, xtrain, ytrain)
                
            with open(f'../result_data/csv_data/{self.log_name}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['train loss', 'train acc', 'valid loss', 'valid acc'])
        else: 
            print('Retraining current task')

        self.shape_out = self.model.layers[-1].shape_out
        self.cur_task = len(self.shape_out)-1

        self.lamb = self.lambs[self.cur_task-1]
        print('lambda', self.lamb)


        self.train_phase(t, xtrain, ytrain, xvalid, yvalid, train_loader, valid_loader, prune_loader, True)
        if self.check_point['phase'] == 2 or self.check_point['phase'] == 3:
            self.check_point = {'model':self.model, 'phase':3}
            torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        # self.plot_neurons_importance()
        # if t > 1:
        #   self.prune_previous(t, xtrain, ytrain)
        self.prune(t, xtrain, ytrain, prune_loader)


        self.check_point = {'model':self.model, 'phase':2, 'optimizer':self._get_optimizer(self.lr), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.train_phase(t, xtrain, ytrain, xvalid, yvalid, train_loader, valid_loader, prune_loader, False)

        # with torch.no_grad():
        #   for m in self.model.DM:
        #       norm = m.norm_in()
        #       m.norm.append(norm.clone())
        #       print(m.norm[-1])

        # self.plot_neurons_importance()

        self.check_point = {'model':self.model, 'phase':3}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
        # for m in self.model.DM:
        #   print(m.weight.data)
        

    def train_phase(self, t, xtrain, ytrain, xvalid, yvalid, train_loader, valid_loader, prune_loader, squeeze):

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)

        train_loss,train_acc=self.eval(t,xtrain,ytrain,prune_loader)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t,xvalid,yvalid,valid_loader)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        if self.check_point['phase'] == 3:
            lr = self.lr
            patience = self.lr_patience
            self.optimizer = self._get_optimizer(lr)
            start_epoch = 0
            phase = 1
        else:
            lr = self.check_point['lr']
            patience = self.check_point['patience']
            self.optimizer = self.check_point['optimizer']
            phase = self.check_point['phase']
            start_epoch = self.check_point['epoch'] + 1


        if phase == 1:
            squeeze = True
            nepochs = self.nepochs
        else:
            squeeze = False
            nepochs = self.nepochs

        if squeeze:
            best_acc = train_acc
        else:
            best_acc = valid_acc
    
        try:
            for e in range(start_epoch, nepochs):
            # Train
                # if e % 10 == 0:
                #   lip = naive_lip(self.model, 1000)
                #   print('lip const:', lip)
                clock0=time.time()
                # self.optimizer = self._get_optimizer(lr)
                self.train_epoch(t, xtrain, ytrain, train_loader, squeeze)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(t, xtrain, ytrain, prune_loader)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, xvalid, yvalid, valid_loader)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                
                # Adapt lr
                if squeeze:
                    if train_acc >= best_acc:
                        best_acc = train_acc
                        self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'phase':phase, 'epoch':e, 'lr':lr, 'patience':patience}
                        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                        print(' *', end='')
                        patience = self.lr_patience

                    # else:
                    #   patience -= 1
                    #   if patience <= 0:
                    #       lr /= self.lr_factor
                    #       print(' lr={:.1e}'.format(lr), end='')
                    #       if lr < self.lr_min:
                    #           print()
                    #           break
                                
                    #       patience = self.lr_patience
                    #       self.optimizer = self._get_optimizer(lr)

                else:
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'phase':phase, 'epoch':e, 'lr':lr, 'patience':patience}
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

                # if e + 1 == 50 or e + 1 == 80:
                #   lr *= 0.1

                print()
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
        self.model.to(device)

    def train_epoch(self, t, x, y, data_loader, squeeze=True):
        self.model.train()

        if x is not None:
            r=np.arange(x.size(0))
            np.random.shuffle(r)
            r=torch.LongTensor(r)

            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=x[b].to(device)
                targets=y[b].to(device)

                if t is None:
                    # outputs = self.model.forward(images, t=-1)

                    outputs = []
                    for task in range(1, self.cur_task + 1):
                        output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                        output = output - output.mean(-1).view(-1, 1)
                        outputs.append(output)

                    outputs = torch.cat(outputs, dim=-1)
                else:
                    outputs = self.model.forward(images, t=t)
                    outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                loss = self.ce(outputs, targets)

                if squeeze:
                    loss += self.model.group_lasso_reg() * self.lamb

                self.optimizer.zero_grad()
                loss.backward()
                # if t > 1:
                #   self.model.track_gradient(len(b))
                # Restrict layer gradients in backprop
                self.model.restrict_gradients()

                # if self.optim == 'SGD':
                #   torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

                self.optimizer.step()

        else:
            for images, targets in data_loader:
                images=images.to(device)
                targets=targets.to(device)

                if t is None:
                    # outputs = self.model.forward(images, t=-1)

                    outputs = []
                    for task in range(1, self.cur_task + 1):
                        output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                        output = output - output.mean(-1).view(-1, 1)
                        outputs.append(output)

                    outputs = torch.cat(outputs, dim=-1)
                else:
                    outputs = self.model.forward(images, t=t)
                    outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                loss = self.ce(outputs, targets)

                if squeeze:
                    loss += self.model.group_lasso_reg() * self.lamb

                self.optimizer.zero_grad()
                loss.backward()
                # if t > 1:
                #   self.model.track_gradient(len(b))
                # Restrict layer gradients in backprop
                self.model.restrict_gradients()

                # if self.optim == 'SGD':
                #   torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

                self.optimizer.step()


    def eval(self, t, x, y, data_loader, masks=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        if x is not None:
            r = np.arange(x.size(0))
            r = torch.LongTensor(r)

            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=x[b].to(device)
                targets=y[b].to(device)
                    
                if masks is not None:
                    if t is None:
                        # outputs = self.model.forward_mask(images, masks, t=-1)

                        outputs = []
                        for task in range(1, self.cur_task):
                            output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                            output = output - output.mean(-1).view(-1, 1)
                            outputs.append(output)

                        output = self.model.forward_mask(images, masks, t=-1)[:, self.shape_out[-2]:self.shape_out[-1]]
                        output = output - output.mean(-1).view(-1, 1)
                        outputs.append(output)
                        outputs = torch.cat(outputs, dim=-1)
                    else:
                        outputs = self.model.forward_mask(images, masks, t=t)
                        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                    # outputs = self.model.forward_mask(images, masks, t=-1)

                else:
                    if t is None:
                        # outputs = self.model.forward(images, t=-1)

                        outputs = []
                        for task in range(1, self.cur_task + 1):
                            output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                            output = output - output.mean(-1).view(-1, 1)
                            outputs.append(output)
                        outputs = torch.cat(outputs, dim=-1)
                        print(outputs)
                    else:
                        outputs = self.model.forward(images, t=t)
                        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                    # outputs = self.model.forward(images, t=-1)
                        
                loss=self.ce(outputs,targets)
                values,indices=outputs.max(1)
                hits=(indices==targets).float()

                total_loss+=loss.data.cpu().numpy()*len(b)
                total_acc+=hits.sum().data.cpu().numpy()
                total_num+=len(b)

        else:
            for images, targets in data_loader:
                images=images.to(device)
                targets=targets.to(device)
                    
                if masks is not None:
                    if t is None:
                        # outputs = self.model.forward_mask(images, masks, t=-1)

                        outputs = []
                        for task in range(1, self.cur_task):
                            output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                            output = output - output.mean(-1).view(-1, 1)
                            outputs.append(output)

                        output = self.model.forward_mask(images, masks, t=-1)[:, self.shape_out[-2]:self.shape_out[-1]]
                        output = output - output.mean(-1).view(-1, 1)
                        outputs.append(output)
                        outputs = torch.cat(outputs, dim=-1)
                    else:
                        outputs = self.model.forward_mask(images, masks, t=t)
                        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                    # outputs = self.model.forward_mask(images, masks, t=-1)

                else:
                    if t is None:
                        # outputs = self.model.forward(images, t=-1)

                        outputs = []
                        for task in range(1, self.cur_task + 1):
                            output = self.model.forward(images, t=task)[:, self.shape_out[task-1]:self.shape_out[task]]
                            output = output - output.mean(-1).view(-1, 1)
                            outputs.append(output)
                        outputs = torch.cat(outputs, dim=-1)
                        print(outputs)
                    else:
                        outputs = self.model.forward(images, t=t)
                        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

                    # outputs = self.model.forward(images, t=-1)
                        
                loss=self.ce(outputs,targets)
                values,indices=outputs.max(1)
                hits=(indices==targets).float()

                total_loss+=loss.data.cpu().numpy()*len(targets)
                total_acc+=hits.sum().data.cpu().numpy()
                total_num+=len(targets)

        return total_loss/total_num,total_acc/total_num


    def prune(self, t, x, y, data_loader, thres=0.0):

        loss,acc=self.eval(t,x,y,data_loader,None)
        loss, acc = round(loss, 3), round(acc, 3)
        print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
        # pre_prune_acc = acc
        pre_prune_loss = loss
        prune_ratio = np.ones(len(self.model.DM)-1)
        step = 0
        masks = [None for m in self.model.DM[:-1]]

        # Dynamic expansion
        while sum(prune_ratio) > 0.0:
            t1 = time.time()
            fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 2))
            # fig1, axs1 = plt.subplots(1, 3, figsize=(9, 2))
            mask_in = torch.ones(self.model.DM[0].in_features).bool().cuda()
            print('Pruning ratio:', end=' ')
            for i in range(0, len(self.model.DM)-1):
                m = self.model.DM[i]
                mask_out = torch.ones(m.shape_out[-1]).bool().to(device)
                # remove neurons
                m.squeeze(mask_in, mask_out)
                mask_in = torch.ones(m.in_features).bool().cuda()
                norm = m.norm_in() * m.bn_norm()
                if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                    norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
                                                                m.shape_out[-1]-m.shape_out[-2], 
                                                                self.model.smid, self.model.smid))

                    # norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
                    #                                           m.shape_out[-1], 
                    #                                           self.model.smid, self.model.smid))
                else:
                    norm *= self.model.DM[i+1].norm_out()

                # norm = norm[m.shape_out[-2]:]
                # m.omega = 1/norm
                low, high = 0, norm.shape[0]

                #visualize
                # if i in [4, 5, 6]:
                #   axs1[i-4].hist(norm.detach().cpu().numpy(), bins=100)
                #   axs1[i-4].set_title(f'layer {i+1}')

                axs[i].hist(norm.detach().cpu().numpy(), bins=100)
                axs[i].set_title(f'layer {i+1}')


                if norm.shape[0] != 0:
                    values, indices = norm.sort(descending=True)
                    while True:
                        k = (high+low)//2
                        # if k == norm.shape[0]:
                        #   masks[i][m.shape_out[-2]:] = True
                        # else:
                        # Select top-k biggest norm
                        mask_out[m.shape_out[-2]:] = (norm>values[k])
                        # mask_out = (norm>values[k])
                        masks[i] = mask_out
                        loss, acc = self.eval(t, x, y, data_loader, masks)
                        loss, acc = round(loss, 3), round(acc, 3)
                        # post_prune_acc = acc
                        post_prune_loss = loss
                        if post_prune_loss <= pre_prune_loss:
                        # if pre_prune_acc <= post_prune_acc:
                            # k is satisfy, try smaller k
                            high = k
                            pre_prune_loss = post_prune_loss
                        else:
                            # k is not satisfy, try bigger k
                            low = k

                        if k == (high+low)//2:
                            break

                if high == norm.shape[0]:
                    # not found any k satisfy, keep all neurons
                    mask_out[m.shape_out[-2]:] = True
                    # mask_out = True
                else:
                    # found k = high is the smallest k satisfy
                    mask_out[m.shape_out[-2]:] = (norm>values[high])
                    # mask_out = (norm>values[high])

                masks[i] = None
                # remove neurons 
                m.squeeze(mask_in, mask_out)

                if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                    mask_in = mask_out.view(-1,1,1).expand(mask_out.size(0),self.model.smid,self.model.smid).contiguous().view(-1)
                else:
                    mask_in = mask_out

                mask_count = int(sum(mask_out[m.shape_out[-2]:]))
                total_count = int(np.prod(mask_out[m.shape_out[-2]:].shape))
                if total_count == 0:
                    prune_ratio[i] = 0.0
                else:
                    prune_ratio[i] = 1.0 - mask_count/total_count
                print('{:.3f}'.format(prune_ratio[i]), end=' ')

            mask_out = torch.ones(self.model.DM[-1].out_features).bool().cuda()
            self.model.DM[-1].squeeze(mask_in, mask_out)
            print('| Time={:5.1f}ms'.format((time.time()-t1)*1000))
            # self.model.squeeze(masks)
            fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}.pdf', bbox_inches='tight')
            # fig1.savefig(f'../result_data/images/{self.file_name}_task{t}_step_{step}_half.pdf', bbox_inches='tight')
            # plt.show()
            step += 1
            # break
        loss,acc=self.eval(t,x,y,data_loader,None)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)


    def prune_previous(self, t, x, y):
        self.get_grad(t, x, y)
        loss,acc=self.eval(t,x,y,None)
        print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
        loss, acc = round(loss, 3), round(acc, 3)
        # pre_prune_acc = acc
        pre_prune_loss = loss
        prune_ratio = np.ones(len(self.model.DM)-1)
        pre_count = 0
        while True:
            t1 = time.time()
            mask_count = 0
            print('Previous use:', end=' ')
            fig, axs = plt.subplots(2, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 3))
            for i in range(0, len(self.model.DM)-1):
                m = self.model.DM[i]
                norm = m.grad_in[:m.shape_out[-2]] + m.grad_out[:m.shape_out[-2]]
                # print(norm)
                # print(m.grad_in[:m.shape_out[-2]])
                # print(m.grad_out[:m.shape_out[-2]])
                low, high = 0, norm.shape[0]
                mask_pre_temp = m.mask_pre[t]
                if norm.shape[0] != 0:
                    values, indices = norm.sort(descending=True)
                    while True:
                        k = (high+low)//2
                        # sellect top-k smallest
                        m.mask_pre[t] = (norm>values[k])
                        loss, acc = self.eval(t, x, y, None)
                        loss, acc = round(loss, 3), round(acc, 3)
                        post_prune_loss = loss
                        # print(post_prune_loss)
                        if post_prune_loss <= pre_prune_loss:
                            # k is satisfy, try smaller k
                            high = k
                            pre_prune_loss = post_prune_loss
                        else:
                            # k is not satisfy, try bigger k
                            low = k

                        if k == (high+low)//2:
                            break

                if high != norm.shape[0]:
                    # found k = high is the smallest k satisfy
                    m.mask_pre[t] = (norm>values[high])
                else:
                    m.mask_pre[t] = mask_pre_temp

                print(norm[m.mask_pre[t]])
                mask_pre = torch.ones(m.out_features).bool().cuda()
                mask_pre[:m.shape_out[-2]] = m.mask_pre[t]
                mask_pre = torch.ones(m.out_features).bool().cuda() ^ mask_pre
                # print(mask_pre)
                temp = torch.ones(m.in_features).cuda()
                temp[m.shape_in[-2]:] = 0

                if isinstance(m, DynamicLinear):
                    temp = temp.view(1, -1)
                else:
                    temp = temp.view(1, -1, 1, 1)

                m.weight.data[mask_pre] *= temp
                # print(m.weight.data[mask_pre].shape)
                # print(temp.shape)
                if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                    mask_pre = mask_pre.view(-1,1,1).expand(mask_pre.size(0),self.model.smid,self.model.smid).contiguous().view(-1)

                temp = torch.ones(self.model.DM[i+1].out_features).cuda()
                temp[self.model.DM[i+1].shape_out[-2]:] = 0
                # print(self.model.DM[i+1].weight.data[:, mask_pre].shape)
                # print(temp.shape)
                if isinstance(self.model.DM[i+1], DynamicLinear):
                    temp = temp.view(-1, 1)
                else:
                    temp = temp.view(-1, 1, 1, 1)
                
                # m.strength_in = np.prod(m.weight[m.shape_out[-2]:].shape)
                # m.strength_out = np.prod(m.weight[:, m.shape_in[-2]:].shape)

                axs[0][i].hist(norm.detach().cpu().numpy(), bins=100)
                axs[0][i].set_title(f'layer {i+1}')

                axs[1][i].hist((norm*m.mask_pre[t]).detach().cpu().numpy(), bins=100)
                axs[1][i].set_title(f'layer {i+1}')
                # remove neurons 
                mask_count += sum(m.mask_pre[t].int()).item()
                print('{}/{}'.format(sum(m.mask_pre[t].int()).item(), norm.numel()), end=' ')

            print('| Time={:5.1f}ms'.format((time.time()-t1)*1000))
            # plt.show()
            break
            # if mask_count == pre_count:
            #   break
            pre_count = mask_count

        loss,acc=self.eval(t,x,y,None)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

    def get_grad(self, t, x, y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)

            outputs = self.model.forward(images, t=t)
            outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
            loss = self.ce(outputs, targets) #+ self.model.group_lasso_reg() * self.lamb

            self.optimizer.zero_grad()
            loss.backward()

            self.model.track_gradient(len(b))
            # for i, m in enumerate(self.model.DM[:-1]):
            #   m.grad_weight += m.weight.data[:m.shape_out[-2], :m.shape_in[-2]] * m.weight.grad.data[:m.shape_out[-2], :m.shape_in[-2]] * len(b)
            #   m.grad_bias += m.bias.data[:m.shape_out[-2]] * m.bias.grad.data[:m.shape_out[-2]] * len(b)

        # for m in self.model.DM[:-1]:

            # m.mask_pre[t] = (m.grad_in[:m.shape_out[-2]] <= 0) * (m.grad_out[:m.shape_out[-2]] <= 0)
            # print(m.mask_pre[t].sum(), m.mask_pre[t].numel())
            # m.weight_pre[t][:m.shape_out[-2], :m.shape_in[-2]] = m.grad_weight <= 0
            # m.bias_pre[t][:m.shape_out[-2]] = m.grad_bias <= 0
            # print(m.weight_pre[t][:m.shape_out[-2], :m.shape_in[-2]].sum(), m.weight_pre[t][:m.shape_out[-2], :m.shape_in[-2]].numel())

        # fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 2))

        # for i, m in enumerate(self.model.DM[:-1]):
        #   axs[i].hist(m.S[:m.shape_out[-2]].detach().cpu().numpy(), bins=100)
        #   axs[i].set_title(f'layer {i+1}')

        # plt.show()


    def predict_soft_label(self, x):
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        soft_label = [[] for t in range(self.cur_task)]
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)

            for t in range(1, self.cur_task):
                outputs = self.model.forward(images, t=t)
                outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]
                soft_label[t].append(outputs.detach().cpu())

        for t in range(1, self.cur_task):
            soft_label[t] = torch.cat(soft_label[t], dim=0)
            print(soft_label[t].shape)

        return soft_label

    def plot_neurons_importance(self):
        fig, axs = plt.subplots(1, len(self.model.DM), figsize=(3*len(self.model.DM)-3, 2))
        for i in range(0, len(self.model.DM)-1):
            m = self.model.DM[i]
            norm = m.norm_in() * m.bn_norm()
            if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                # norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
                #                                           m.shape_out[-1]-m.shape_out[-2], 
                #                                           self.model.smid, self.model.smid))
                norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
                                                            m.shape_out[-1], 
                                                            self.model.smid, self.model.smid))
            else:
                norm *= self.model.DM[i+1].norm_out()


            axs[i].hist(norm.detach().cpu().numpy(), bins=100)
            axs[i].set_title(f'layer {i+1}')

        norm = self.model.DM[-1].norm_in()
        print(norm)
        axs[-1].hist(norm.detach().cpu().numpy(), bins=100)
        plt.show()

