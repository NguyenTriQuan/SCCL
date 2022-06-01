import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


class MLP(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, split = True, notMNIST=False,mul=1):
        super().__init__()

        ncha,size,_=inputsize
        unitN=int(400*mul)
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla=taskcla
        self.split = split
        self.fc1 = BayesianLinear(28*28, unitN,ratio=ratio)
        self.fc2 = BayesianLinear(unitN, unitN,ratio=ratio)
        # self.fc5 = BayesianLinear(unitN, unitN,ratio=ratio)
        # self.fc6 = BayesianLinear(unitN, unitN,ratio=ratio)
        # self.fc7 = BayesianLinear(unitN, unitN,ratio=ratio)
        
        if notMNIST:
            self.fc3=BayesianLinear(unitN,unitN,ratio=ratio)
            self.fc4=BayesianLinear(unitN,unitN,ratio=ratio)
        self.last=torch.nn.ModuleList()
        
        if split:
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))
        
        else:
            self.fc3 = BayesianLinear(unitN, taskcla[0][1],ratio=ratio)
                
        
    def forward(self, t, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x, sample))
        x = F.relu(self.fc2(x, sample))
        # x = F.relu(self.fc5(x, sample))
        # x = F.relu(self.fc6(x, sample))
        # x = F.relu(self.fc7(x, sample))
        if self.notMNIST:
            x=F.relu(self.fc3(x, sample))
            x=F.relu(self.fc4(x, sample))
        
        if self.split:
            y = self.last[t](x)
            
        else:
            x = self.fc3(x, sample)
            y = F.log_softmax(x, dim=1)
        
        return y



class VGG8(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, mul=1):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,int(32*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = BayesianConv2D(int(32*mul),int(32*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = BayesianConv2D(int(32*mul),int(64*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = BayesianConv2D(int(64*mul),int(64*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = BayesianConv2D(int(64*mul),int(128*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = BayesianConv2D(int(128*mul),int(128*mul),kernel_size=3, padding=1, ratio=ratio)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = BayesianConv2D(128,128,kernel_size=3, padding=1, ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = BayesianLinear(int(s*s*128*mul),int(256*mul), ratio = ratio)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(int(256*mul),n))
        self.relu = torch.nn.ReLU()

    def forward(self, t, x, sample=False):
        h=self.relu(self.conv1(x,sample))
        h=self.relu(self.conv2(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h,sample))
        h=self.relu(self.conv4(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h,sample))
        h=self.relu(self.conv6(h,sample))
#         h=self.relu(self.conv7(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h,sample)))
        y = self.last[t](h)
        
        return y


class Alexnet(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, mul=1):
        super().__init__()
        print("haha")
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,64,kernel_size=size//8, ratio=ratio)
        s = compute_conv_output_size(size,size//8) # 32
        # self.conv2 = BayesianConv2D(32,32,kernel_size=3, padding=1, ratio=ratio)
        # s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = BayesianConv2D(64,128,kernel_size=size//10, ratio=ratio)
        s = compute_conv_output_size(s,size//10) # 16
        # self.conv4 = BayesianConv2D(64,64,kernel_size=3, padding=1, ratio=ratio)
        # s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = BayesianConv2D(128,256,kernel_size=2, ratio=ratio)
        s = compute_conv_output_size(s,2) # 8
        # self.conv6 = BayesianConv2D(128,128,kernel_size=3, padding=1, ratio=ratio)
        # s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv7 = BayesianConv2D(128,128,kernel_size=3, padding=1, ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = BayesianLinear(s*s*256,2048, ratio = ratio)
        self.fc2 = BayesianLinear(2048,2048, ratio = ratio)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))
        self.relu = torch.nn.ReLU()

    def forward(self, t, x, sample=False):
        h=self.relu(self.conv1(x,sample))
        # h=self.relu(self.conv2(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h,sample))
        # h=self.relu(self.conv4(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h,sample))
        # h=self.relu(self.conv6(h,sample))
#         h=self.relu(self.conv7(h,sample))
        h=self.drop2(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h,sample)))
        h = self.drop2(self.relu(self.fc2(h,sample)))
        
        return self.last[t](h)