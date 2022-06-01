import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


class Appr(object):

    def __init__(self,model,data_name,nepochs=100,
        sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,
        clipgrad=100, optim='SGD'):
        self.model=model.to(device)
        self.data_name = data_name
        self.name = '{}_{}'.format(self.model.name, self.data_name)

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.optim = optim

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            # return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, valid_loader):

        # tasks = set(ytrain.cpu().numpy())
        # for t in tasks:
        #     self.model.add_task(t)

        # ytrain = torch.stack([(ytrain==t)*1.0 for t in self.model.tasks], dim=-1)
        # ytrain = torch.argmax(ytrain, dim=-1).long()
        # yvalid = torch.stack([(yvalid==t)*1.0 for t in self.model.tasks], dim=-1)
        # yvalid = torch.argmax(yvalid, dim=-1).long()

        self.model.to(device)
        best_acc = -np.inf
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        # Loop epochs
        for e in range(self.nepochs):
            if e + 1 == 50 or e + 1 == 80:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            clock0=time.time()
            
            self.train_epoch(train_loader)

            clock1=time.time()
            train_loss,train_acc=self.eval(train_loader)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                e+1,1000*(clock1-clock0),
                1000*(clock2-clock1),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(valid_loader)
            print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model,'../trained_model/%s.model'%self.name)
                print(' *', end='')
            print()


        # Restore best
        self.model = torch.load('../trained_model/%s.model'%self.name)
        self.model.to(device)
        

    def train_epoch(self,train_loader):
        self.model.train()

        for images, targets in train_loader:

            images=images.to(device)
            targets=targets.to(device)

            # Forward current model
            outputs = self.model.forward(images)
            
            loss=self.ce(outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def eval(self,valid_loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        for images, targets in valid_loader:

            images=images.to(device)
            targets=targets.to(device)
            
            # Forward
            outputs = self.model.forward(images)
                
            loss=self.ce(outputs,targets)
            values,indices=outputs.max(1)
            hits=(indices==targets).float()#*(values>0.5).float()

            total_loss+=loss.data.cpu().numpy()*images.size(0)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=images.size(0)

        return total_loss/total_num,total_acc/total_num


class DynamicAppr(object):

    def __init__(self,model,data_name,is_bayesian=False,lamb1=1e-3,lamb2=1e-3,
        thres=1e-2,alpha=None,phase=2,dropout=0,squeeze=True,nepochs=100,
        sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,
        clipgrad=100, optim='Adam'):
        self.model=model.to(device)
        self.data_name = data_name
        self.name = '{}_{}'.format(self.model.name, self.data_name)
        self.is_bayesian = is_bayesian

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.optim = optim
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.thres = thres
        self.alpha = alpha
        self.phase = phase
        self.dropout = dropout
        self.squeeze = squeeze

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, xtrain, ytrain, xvalid, yvalid):

        # tasks = set(ytrain.cpu().numpy())
        # for t in tasks:
        #     self.model.add_task(t)

        # ytrain = torch.stack([(ytrain==t)*1.0 for t in self.model.tasks], dim=-1)
        # ytrain = torch.argmax(ytrain, dim=-1).long()
        # yvalid = torch.stack([(yvalid==t)*1.0 for t in self.model.tasks], dim=-1)
        # yvalid = torch.argmax(yvalid, dim=-1).long()

        self.model.to(device)
        best_acc = -np.inf
        lr = self.lr
        patience = self.lr_patience
        # self.optimizer = self._get_optimizer(lr)
        best_mask = np.inf
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            # if e < 10:
            #     self.lamb1 = 0
            #     self.lamb2 = 0
            # else:
            #     self.lamb1 = self.lamb1_
            #     self.lamb2 = self.lamb2_

            if self.phase == 1:
                break
            self.optimizer = self._get_optimizer(lr)

            clock0=time.time()
            
            num_batch = xtrain.size(0)
            
            self.train_epoch(xtrain,ytrain,squeeze=True,dropout=self.dropout)

            clock1=time.time()
            train_loss,train_acc=self.eval(xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                e+1,1000*(clock1-clock0),
                1000*(clock2-clock1),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
            
            print()

        torch.save(self.model,'./trained_model/%s.model'%self.name)
        valid_loss,valid_acc=self.eval(xvalid,yvalid)
        print('Post Prune Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))
        best_acc = valid_acc
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        for e in range(self.nepochs):
            if self.phase == 1 and self.squeeze:
                self.optimizer = self._get_optimizer(lr)
            # Train
            clock0=time.time()
            
            num_batch = xtrain.size(0)
            
            if self.phase == 1:
                self.train_epoch(xtrain,ytrain,squeeze=self.squeeze,dropout=self.dropout)
            else:
                self.train_epoch(xtrain,ytrain,squeeze=False,dropout=self.dropout)

            
            clock1=time.time()
            train_loss,train_acc=self.eval(xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                e+1,1000*(clock1-clock0),
                1000*(clock2-clock1),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
            
            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model,'./trained_model/%s.model'%self.name)
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

        # Restore best
        self.model = torch.load('./trained_model/%s.model'%self.name)
        self.model.to(device)
        

    def train_epoch(self,x,y,squeeze,dropout):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)
        # print(self.model.mask1.sum(), self.model.mask2.sum())
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)

            # Forward current model
            outputs = self.model.forward(images,dropout)

            if squeeze:
                if self.alpha is None:
                    loss=self.ce(outputs,targets) + self.model.group_lasso_reg(self.lamb1, self.lamb2)
                else:
                    loss=self.ce(outputs,targets) + self.model.sparse_group_lasso_reg(self.lamb1, self.lamb2, self.alpha)                    
            else:
                loss=self.ce(outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if squeeze:
            self.model.squeeze(self.thres)
            for n, p in self.model.named_parameters():
                if 'bias' in n:
                    print(p.shape[0], end=' ')
            print()



    def eval(self,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)
            
            # Forward
            outputs = self.model.forward(images)
                
            loss=self.ce(outputs,targets)
            values,indices=outputs.max(1)
            hits=(indices==targets).float()#*(values>0.5).float()

            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num


    def compute_model_size(self):
        count=0
        for p in self.model.parameters():
            count+=np.prod(p.size())
        print('-'*100)
        print('Num parameters = %s'%(human_format(count)))
        print('-'*100)
        return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])