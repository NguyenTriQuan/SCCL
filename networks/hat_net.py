import sys
import torch
import torch.nn as nn
from utils import *


class MLP(torch.nn.Module):

    def __init__(self,inputsize,taskcla, split = True, notMNIST = False, mul=1):
        super(MLP,self).__init__()

        ncha,size,_=inputsize
        unitN=int(400*mul)
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla=taskcla
        self.split = split
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,unitN)
        self.efc1=torch.nn.Embedding(len(self.taskcla),unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)
        self.efc2=torch.nn.Embedding(len(self.taskcla),unitN)
        if notMNIST:
            self.fc3=torch.nn.Linear(unitN,unitN)
            self.efc3=torch.nn.Embedding(len(self.taskcla),unitN)
            self.fc4=torch.nn.Linear(unitN,unitN)
            self.efc4=torch.nn.Embedding(len(self.taskcla),unitN)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(unitN,n))
        
        self.gate=torch.nn.Sigmoid()
        
        
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""
        self.layers = [self.fc1, self.fc2]
        self.embeddings = [self.efc1, self.efc2]
        self.base = [400, 400]
        self.masks = [[] for _ in range(2)]
        return

    def forward(self,t,x,s=1):
        # Gates
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        
        # Gated
        h=self.drop2(x.view(x.size(0),-1))
        h=self.drop1(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop1(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        if self.notMNIST:
            gfc3=self.gate(s*self.efc3(t))
            gfc4=self.gate(s*self.efc4(t))
            h=self.drop1(self.relu(self.fc3(h)))
            h=h*gfc3.expand_as(h)
            h=self.drop1(self.relu(self.fc4(h)))
            h=h*gfc4.expand_as(h)
        
        if self.split:
            y=self.last[t](h)
        else:
#             y=self.relu(self.fc3(h))
            y=self.fc3(h)
            
        masks = [gfc1, gfc2]
        if self.notMNIST:
            mask = [gfc1, gfc2, gfc3, gfc4]
        
        return y,masks

    def mask(self,t,s=1):
        try:
            return [mask[t] for mask in self.masks]
        except:
            pass
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        if self.notMNIST:
            gfc3=self.gate(s*self.efc3(t))
            gfc4=self.gate(s*self.efc4(t))
            return [gfc1, gfc2, gfc3, gfc4]

        return [gfc1, gfc2]
        

    def get_view_for(self,n,masks):
        if self.notMNIST:
            gfc1,gfc2,gfc3,gfc4=masks
        else:
            gfc1,gfc2=masks
        
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='fc3.weight':
            return gfc2.data.view(1,-1).expand_as(self.fc3.weight)
        if self.notMNIST:
            if n=='fc3.weight':
                post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
                pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
                return torch.min(post,pre)
            elif n=='fc3.bias':
                return gfc3.data.view(-1)
            elif n=='fc4.weight':
                post=gfc4.data.view(-1,1).expand_as(self.fc4.weight)
                pre=gfc3.data.view(1,-1).expand_as(self.fc4.weight)
                return torch.min(post,pre)
            elif n=='fc4.bias':
                return gfc4.data.view(-1)

        
        return None



class VGG8(torch.nn.Module):

    def __init__(self,inputsize,taskcla,mul=1):
        super(VGG8,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.mul = mul

        self.c1 = nn.Conv2d(ncha,int(32*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.c2 = nn.Conv2d(int(32*mul),int(32*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.c3 = nn.Conv2d(int(32*mul),int(64*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.c4 = nn.Conv2d(int(64*mul),int(64*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.c5 = nn.Conv2d(int(64*mul),int(128*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.c6 = nn.Conv2d(int(128*mul),int(128*mul),kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.c7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(int(s*s*128*mul),int(256*mul)) # 2048
        self.drop1=torch.nn.Dropout(0.25)
        self.drop2=torch.nn.Dropout(0.5)
        
        self.smid=s
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(int(256*mul),n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),int(32*mul))
        self.ec2=torch.nn.Embedding(len(self.taskcla),int(32*mul))
        self.ec3=torch.nn.Embedding(len(self.taskcla),int(64*mul))
        self.ec4=torch.nn.Embedding(len(self.taskcla),int(64*mul))
        self.ec5=torch.nn.Embedding(len(self.taskcla),int(128*mul))
        self.ec6=torch.nn.Embedding(len(self.taskcla),int(128*mul))
#         self.ec7=torch.nn.Embedding(len(self.taskcla),128)
        self.efc1=torch.nn.Embedding(len(self.taskcla),int(256*mul))
        
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.ec4.weight.data.uniform_(lo,hi)
        self.ec5.weight.data.uniform_(lo,hi)
        self.ec6.weight.data.uniform_(lo,hi)
        self.ec7.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        #"""
        self.layers = [self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.fc1]
        self.embeddings = [self.ec1, self.ec2, self.ec3, self.ec4, self.ec5, self.ec6, self.efc1]
        self.base = [32, 32, 64, 64, 128, 128, 256]
        self.masks = [[] for _ in range(7)]
        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
#         gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1=masks
        gc1,gc2,gc3,gc4,gc5,gc6,gfc1=masks
        
        # Gated
        h=self.relu(self.c1(x))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c2(h))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c3(h))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c4(h))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c5(h))
        h=h*gc5.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c6(h))
        h=h*gc6.view(1,-1,1,1).expand_as(h)
#         h=self.relu(self.c7(h))
#         h=h*gc7.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=h.view(x.shape[0],-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        y=self.last[t](h)
        
        return y,masks

    def mask(self,t,s=1):
        try:
            return [mask[t] for mask in self.masks]
        except:
            pass
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        gc5=self.gate(s*self.ec5(t))
        gc6=self.gate(s*self.ec6(t))
#         gc7=self.gate(s*self.ec7(t))
        gfc1=self.gate(s*self.efc1(t))
#         return [gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1]
        return [gc1,gc2,gc3,gc4,gc5,gc6,gfc1]

    def get_view_for(self,n,masks):
#         gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1=masks
        gc1,gc2,gc3,gc4,gc5,gc6,gfc1=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc6.data.view(-1,1,1).expand((self.ec6.weight.size(1),
                                              self.smid,
                                              self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        elif n=='c4.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.c4.weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.c4.weight)
            return torch.min(post,pre)
        elif n=='c4.bias':
            return gc4.data.view(-1)
        elif n=='c5.weight':
            post=gc5.data.view(-1,1,1,1).expand_as(self.c5.weight)
            pre=gc4.data.view(1,-1,1,1).expand_as(self.c5.weight)
            return torch.min(post,pre)
        elif n=='c5.bias':
            return gc5.data.view(-1)
        elif n=='c6.weight':
            post=gc6.data.view(-1,1,1,1).expand_as(self.c6.weight)
            pre=gc5.data.view(1,-1,1,1).expand_as(self.c6.weight)
            return torch.min(post,pre)
        elif n=='c6.bias':
            return gc6.data.view(-1)
#         elif n=='c7.weight':
#             post=gc7.data.view(-1,1,1,1).expand_as(self.c7.weight)
#             pre=gc6.data.view(1,-1,1,1).expand_as(self.c7.weight)
#             return torch.min(post,pre)
#         elif n=='c7.bias':
#             return gc7.data.view(-1)
        return None



class Alexnet(torch.nn.Module):

    def __init__(self,inputsize,taskcla,mul=1):
        super(Alexnet,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),128)
        self.ec3=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),2048)
        self.efc2=torch.nn.Embedding(len(self.taskcla),2048)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""
        self.layers = [self.c1, self.c2, self.c3, self.fc1, self.fc2]
        self.embeddings = [self.ec1, self.ec2, self.ec3, self.efc1, self.efc2]
        self.base = [64, 128, 256, 2048, 2048]
        self.masks = [[] for _ in range(5)]
        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gfc1,gfc2=masks
        # Gated
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        y=self.last[t](h)
        
        return y,masks

    def mask(self,t,s=1):
        try:
            return [mask[t] for mask in self.masks]
        except:
            pass
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.ec3.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        return None


class HATLayer(nn.Module):
    """docstring for HATLayer"""
    def __init__(self, tasknum, num_features):
        super(HATLayer, self).__init__()
        self.tasknum = tasknum
        self.embedding = nn.Embedding(tasknum, num_features)
        self.gate = nn.Sigmoid()

    def forward(self, t, x, s):
        mask = self.gate(s*self.embedding(t))
        if len(x.shape) == 4:
            mask = mask.view(1,-1,1,1).expand_as(x)
        else:
            mask = mask.expand_as(h)

        return x*mask, mask

class VGG16(nn.Module):

    def __init__(self, input_size, taskcla, mul=1, batch_norm=True):
        super(Net, self).__init__()
        ncha, size, _ = input_size
        self.mul = mul
        s = size

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = ncha
        tasknum = len(taskcla)

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                s = s//2
            else:
                conv2d = nn.Conv2d(in_channels, int(v * mul), kernel_size=3, padding=1, bias=False)
                s = compute_conv_output_size(s,3, padding=1)
                if batch_norm:
                    layers += [HATLayer(conv2d, tasknum, int(v * mul)), nn.BatchNorm2d(int(v * mul)), nn.ReLU(inplace=True)]
                else:
                    layers += [HATLayer(conv2d, tasknum, int(v * mul)), nn.ReLU(inplace=True)]
                in_channels = int(v * mul)

        self.smid = s
        layers += [
            View(-1, int(512*mul)),
            nn.Linear(int(512*mul), int(4096*mul)),
            nn.ReLU(True),
            nn.Linear(int(4096*mul), int(4096*mul)),
            nn.ReLU(True),
        ]

        self.layers = nn.ModuleList(layers)
        for t,n in self.taskcla:
            self.last.append(nn.Linear(int(4096*mul),n))

    def forward(self, t, x, s=1):
        masks = []
        for module in self.layers:
            if isinstance(module, HATLayer):
                x, mask = module(t, x, s)
                masks.append(mask)
            else:
                x= module(x)

        y = self.last[t](x)
        
        return y, masks


    def get_view_for(self,n,masks):
