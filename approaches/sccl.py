import sys, time, os
import math

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
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer
import networks.sccl_net as network
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

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform, ncla=0):

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

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, True)
        if not self.check_point['squeeze']:
            self.check_point = None
            return 

        self.prune(t, train_loader, valid_transform, thres=self.thres)


        self.check_point = {'model':self.model, 'squeeze':False, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
        torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

        self.train_phase(t, train_loader, valid_loader, train_transform, valid_transform, False)

        self.check_point = None
        # self.model.get_params(t-1)
        # for m in self.model.DM:
        #     weight = torch.cat([torch.cat([m.old_weight, m.fwt_weight[t]], dim=0), torch.cat([m.bwt_weight[t], m.weight[t]], dim=0)], dim=1)
        #     norm = weight.norm(2).detach()
        #     m.weight[t].data /= norm
        #     if m.bias:
        #         m.bias[t].data /= norm

        # s_H = self.model.s_H()
        # print('s_H={:.1e}'.format(s_H), end='')
        

    def train_phase(self, t, train_loader, valid_loader, train_transform, valid_transform, squeeze):

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)

        train_loss,train_acc=self.eval(t,train_loader,valid_transform)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(t,valid_loader,valid_transform)
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
                self.train_epoch(t, train_loader, train_transform, squeeze)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(t, train_loader, valid_transform)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
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
        # images = self.train_transforms(images)
        outputs = self.model.forward(images, t=t)
        outputs = outputs[:, self.shape_out[t-1]:self.shape_out[t]]

        loss = self.ce(outputs, targets)

        if squeeze:
            loss += self.model.group_lasso_reg() * self.lamb
                
        self.optimizer.zero_grad()
        # loss.backward() 
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

    def train_epoch(self, t, data_loader, train_transform, squeeze=True):
        self.model.train()
        self.model.get_params(t-1)
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = train_transform(images)
            self.train_batch(t, images, targets, squeeze)


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

        loss,acc=self.eval(t,data_loader,valid_transform)
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

            fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}.pdf', bbox_inches='tight')
            # plt.show()
            loss,acc=self.eval(t,data_loader,valid_transform)
            print('| Post Prune: loss={:.3f}, acc={:5.2f}% | Time={:5.1f}ms |'.format(loss, 100*acc, (time.time()-t1)*1000))

            step += 1
            if sum(prune_ratio) == pre_sum:
                break
            pre_sum = sum(prune_ratio)

        for m in self.model.DM[:-1]:
            m.squeeze()
            m.mask = None
        loss,acc=self.eval(t,data_loader,valid_transform)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

        print('number of neurons:', end=' ')
        for m in self.model.DM:
            print(m.out_features, end=' ')
        print()
        params = self.model.compute_model_size()
        print('num params', params)


    def prune_previous(self, t, x, y):
        # for m in self.model.DM[:-1]:
        #   nn.init.constant_(m.weight[t], 0)
        #   nn.init.constant_(m.fwt_weight[t], 0)
        #   nn.init.constant_(m.bwt_weight[t], 0)

        # m = self.model.DM[-1]
        # nn.init.constant_(m.weight[t], 1)
        # nn.init.constant_(m.fwt_weight[t], 1)
        # nn.init.constant_(m.bwt_weight[t], 1)

        loss,acc=self.eval(t,x,y,None)
        print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
        loss, acc = round(loss, 3), round(acc, 3)
        # pre_prune_acc = acc
        pre_prune_loss = loss
        prune_ratio = np.ones(len(self.model.DM)-1)
        pre_count = 0
        step=0
        masks = [None for m in self.model.DM[:-1]]
        while True:
            t1 = time.time()
            self.get_grad(t, x, y)
            mask_count = 0
            print('Previous use:', end=' ')
            fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 3))
            mask_in = None
            for i in range(0, len(self.model.DM)-1):
                m = self.model.DM[i]
                mask_out = None
                m.squeeze_previous(mask_in, mask_out)
                mask_in = None
                mask_temp = torch.ones(m.weight[-1].shape[0]).float().cuda()
                norm = m.grad_in + m.grad_out
                low, high = 0, norm.shape[0]
                if norm.shape[0] != 0:
                    values, indices = norm.sort(descending=True)
                    # print(values)
                    # while True:
                    #   k = (high+low)//2
                    #   # sellect top-k smallest
                    #   m.mask_pre_out[t] = (norm>values[k])
                    #   loss, acc = self.eval(t, x, y, None)
                    #   loss, acc = round(loss, 3), round(acc, 3)
                    #   post_prune_loss = loss
                    #   # print(post_prune_loss)
                    #   if post_prune_loss <= pre_prune_loss:
                    #       # k is satisfy, try smaller k
                    #       high = k
                    #       pre_prune_loss = post_prune_loss
                    #   else:
                    #       # k is not satisfy, try bigger k
                    #       low = k

                    #   if k == (high+low)//2:
                    #       break

                    losses = []
                    high = len(norm)
                    for k in range(len(norm)):
                        mask_out = (norm>values[k])
                        masks[i] = torch.cat([mask_out.float(), mask_temp])
                        loss, acc = self.eval(t, x, y, masks)
                        loss, acc = round(loss, 3), round(acc, 3)
                        losses.append(loss)
                        if loss <= pre_prune_loss:
                            pre_prune_loss = loss
                            high = k

                    loss, acc = self.eval(t, x, y, None)
                    loss, acc = round(loss, 3), round(acc, 3)
                    losses.append(loss)
                    axs[i].plot(range(len(norm)+1), losses)
                    axs[i].set_xlabel('k')
                    axs[i].set_ylabel('loss')
                    axs[i].set_title(f'layer {i+1}')

                if high != norm.shape[0]:
                    # found k = high is the smallest k satisfy
                    mask_out = (norm>values[high])
                else:
                    mask_out = None

                m.squeeze_previous(mask_in, mask_out)

                if mask_out is None:
                    mask_in = None
                else:
                    if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
                        mask_in = mask_out.view(-1,1,1).expand(mask_out.size(0),self.model.smid,self.model.smid).contiguous().view(-1)
                    else:
                        mask_in = mask_out

                # remove neurons 
                if mask_out is None:
                    mask_count = norm.numel()
                else:
                    mask_count = sum(mask_out.int()).item()
                print('{}/{}'.format(mask_count, norm.numel()), end=' ')


            mask_out = None
            self.model.DM[-1].squeeze_previous(mask_in, mask_out)
            print('| Time={:5.1f}ms'.format((time.time()-t1)*1000))
            plt.show()
            fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}_prune_previous.pdf', bbox_inches='tight')
            step += 1
            break
            # if mask_count == pre_count:
            #   break
            pre_count = mask_count

        loss,acc=self.eval(t,x,y,None)
        print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

    def get_grad(self, t, x, y):
        self.model.train()
        self.model.get_params(t-1)
        self._get_optimizer(lr=None,track_grad=True)
        for m in self.model.DM:
            m.old_weight.requires_grad = True
            m.old_bias.requires_grad = True
            m.old_weight.grad = None
            m.old_bias.grad = None
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        for m in self.model.DM:
            m.grad_in = 0
            m.grad_out = 0
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

        for m in self.model.DM:
            m.old_weight.requires_grad = False
            m.old_bias.requires_grad = False


