import sys
import torch
import torch.nn as nn
from utils import *


class MLP(torch.nn.Module):

    def __init__(self,inputsize,taskcla, mul=1):
        super(MLP,self).__init__()

        ncha,size,_=inputsize
        unitN=400
        self.taskcla=taskcla
        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,int(unitN*mul))
        self.fc2=torch.nn.Linear(int(unitN*mul),int(unitN*mul))
        self.avg_neg = []
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(int(unitN*mul),n))
    
    def forward(self, t, x, avg_act = False):
        h=x.view(x.size(0),-1)
        act1 = self.relu(self.fc1(h))
        h = self.drop(act1)
        act2 = self.relu(self.fc2(h))
        h = self.drop(act2)
        
        y = self.last[t](h)
         

        self.grads={}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook


        if avg_act == True:
            names = [3, 4]
            act = [act1, act2]
            
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        return y


class VGG8(nn.Module):
    def __init__(self, inputsize, taskcla, mul=1):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,int(32*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = nn.Conv2d(int(32*mul),int(32*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(int(32*mul),int(64*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(int(64*mul),int(64*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(int(64*mul),int(128*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(int(128*mul),int(128*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(int(s*s*128*mul),int(256*mul)) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(int(256*mul),n))
        self.relu = torch.nn.ReLU()

    def forward(self, t, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        act2=self.relu(self.conv2(act1))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.conv3(h))
        act4=self.relu(self.conv4(act3))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.conv5(h))
        act6=self.relu(self.conv6(act5))
        h=self.drop1(self.MaxPool(act6))
        h=h.view(x.shape[0],-1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        y = self.last[t](h)
        
        self.grads={}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook
        
        """
        act1=self.conv1(x)
        act2=self.conv2(self.relu(act1))
        h=self.drop1(self.MaxPool(self.relu(act2)))
        act3=self.conv3(h)
        act4=self.conv4(self.relu(act3))
        h=self.drop1(self.MaxPool(self.relu(act4)))
        act5=self.conv5(h)
        act6=self.conv6(self.relu(act5))
#         h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(self.relu(act6)))
        h=h.view(x.shape[0],-1)
        act7 = self.fc1(h)
        h = self.drop2(self.relu(act7))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        """
        
        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]
            
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))
        return y

class Alexnet(nn.Module):
    def __init__(self, inputsize, taskcla, mul=1):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=size//8)
        s = compute_conv_output_size(size,size//8) # 32
        # self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        # s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(64,128,kernel_size=size//10)
        s = compute_conv_output_size(s,size//10) # 16
        # self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        # s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(128,256,kernel_size=2)
        s = compute_conv_output_size(s,2) # 8
        # self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        # s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*256,2048) # 2048
        self.fc2 = nn.Linear(2048,2048)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.avg_neg = []
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))
        self.relu = torch.nn.ReLU()

    def forward(self, t, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        # act2=self.relu(self.conv2(act1))
        h=self.drop1(self.MaxPool(act1))
        act3=self.relu(self.conv3(h))
        # act4=self.relu(self.conv4(act3))
        h=self.drop1(self.MaxPool(act3))
        act5=self.relu(self.conv5(h))
        # act6=self.relu(self.conv6(act5))
        h=self.drop2(self.MaxPool(act5))
        h=h.view(x.shape[0],-1)
        act7 = self.relu(self.fc1(h))
        h = self.drop2(act7)
        act8 = self.relu(self.fc2(h))
        h = self.drop2(act8)
        y = self.last[t](h)
        
        self.grads={}
        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad
            return hook
        
        """
        act1=self.conv1(x)
        act2=self.conv2(self.relu(act1))
        h=self.drop1(self.MaxPool(self.relu(act2)))
        act3=self.conv3(h)
        act4=self.conv4(self.relu(act3))
        h=self.drop1(self.MaxPool(self.relu(act4)))
        act5=self.conv5(h)
        act6=self.conv6(self.relu(act5))
#         h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(self.relu(act6)))
        h=h.view(x.shape[0],-1)
        act7 = self.fc1(h)
        h = self.drop2(self.relu(act7))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        """
        
        if avg_act == True:
            names = [0, 1, 2, 3, 4]
            act = [act1, act3, act5, act7, act8]
            
            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))
        return y