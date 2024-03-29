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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import networks.net as network
args = get_args()

class Appr(object):

    def __init__(self, inputsize, taskcla, args,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100, log_name=None, split=False):
        Net = getattr(network, args.arch)
        self.model = Net(inputsize, taskcla).to(device)

        self.log_name = '{}_{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.arch, 
                                                                                args.seed, args.lr, 
                                                                             args.batch_size, args.nepochs)
        # self.logger = utils.logger(file_name=self.log_name, resume=False, path='../result_data/csv_data/', data_format='csv')
        self.inputsize = inputsize
        self.taskcla = taskcla
        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = args.lr / 100
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience
        self.clipgrad = clipgrad
        self.split = True

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
#         return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def train(self, t, train_loader, valid_loader, train_transform, valid_transform):
        Net = getattr(network, args.arch)
        self.model = Net(self.inputsize, self.taskcla).to(device)
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
            
            self.train_epoch(t, train_loader, train_transform)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(t, train_loader, valid_transform)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0),
                1000*self.sbatch*(clock2-clock1),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t, valid_loader, valid_transform)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            #save log for current task & old tasks at every epoch
            # self.logger.add(epoch=(t*self.nepochs)+e, task_num=t+1, valid_loss=valid_loss, valid_acc=valid_acc)
            
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

        # self.logger.save()
        torch.save(self.model, '../result_data/trained_model/' + self.log_name + '.model')
        return

    def train_epoch(self, t, data_loader, train_transform):
        self.model.train()

        # Loop batches
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = train_transform(images)
            
            # Forward current model
            outputs = self.model.forward(images, t)
            loss=self.ce(outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, t, data_loader, valid_transform):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()


        # Loop batches
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if valid_transform:
                images = valid_transform(images)
            
            # Forward
            outputs = self.model.forward(images, t)
                
            loss=self.ce(outputs,targets)
            _,pred=outputs.max(1)
            hits=(pred==targets).float()

            total_loss+=loss.data.cpu().numpy()*len(targets)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(targets)

        return total_loss/total_num,total_acc/total_num
