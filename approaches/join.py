import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn

args = get_args()

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100, args=None, log_name=None, split=False):
        self.model=model
        self.model_old=model

        self.log_name = '{}_{}_{}_{}_mul_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed, 
                                                                                args.mul, args.lr, 
                                                                             args.batch_size, args.nepochs)
        self.logger = utils.logger(file_name=self.log_name, resume=False, path='../result_data/csv_data/', data_format='csv')

        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.split = True

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.taskcla = [last.out_features for last in self.model.last]
        self.taskcla = [sum(self.taskcla[:i]) for i in range(len(self.taskcla)+1)]
        print(self.taskcla)

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
#         return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def train(self, t, xtrain, ytrain, xvalid, yvalid):
        best_acc = -np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            
            # CUB 200 xtrain_cropped = crop(x_train)
            num_batch = xtrain.size(0)
            
            self.train_epoch(t,xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            #save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t*self.nepochs)+e, task_num=t+1, valid_loss=valid_loss, valid_acc=valid_acc)
            
            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
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
                        if args.conv_net:
                            pass
#                             break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()
        
        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights
        torch.save(self.model, '../result_data/trained_model/' + self.log_name + '.model')

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward current model
            loss = 0
            for task in range(t+1):
                # print(task, ((self.taskcla[task]<=targets)&(targets<self.taskcla[task+1])).shape)
                outputs = self.model.forward(task, images[(self.taskcla[task]<=targets)&(targets<self.taskcla[task+1])])
                loss += self.ce(outputs,targets[(self.taskcla[task]<=targets)&(targets<self.taskcla[task+1])])
            # outputs = self.model.forward(t, images)
            # loss=self.criterion(t,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward
            if self.split:
                output = self.model.forward(t, images)
            else:
                output = self.model.forward(t, images)
                
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
#             total_loss+=loss.data.cpu().numpy()[0]*len(b)
#             total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        return self.ce(output,targets)
