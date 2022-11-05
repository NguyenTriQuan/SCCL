import sys
import torch
import torch.nn.functional as F
from utils import *

class MLP(torch.nn.Module):

    def __init__(self,inputsize,taskcla, split = True, notMNIST = False,mul=1):
        super(MLP,self).__init__()

        ncha,size,_=inputsize
        unitN=int(400*mul)
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla=taskcla
        self.split = split
        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)
        
        if notMNIST:
            self.fc3=torch.nn.Linear(unitN,unitN)
            self.fc4=torch.nn.Linear(unitN,unitN)
        
        if split:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))
        else:
            self.fc3=torch.nn.Linear(unitN,taskcla[0][1])

    def forward(self, x, t):
        h=x.view(x.size(0),-1)
        h=self.drop(F.relu(self.fc1(h)))
        h=self.drop(F.relu(self.fc2(h)))
        if self.notMNIST:
            h=self.drop(F.relu(self.fc3(h)))
            h=self.drop(F.relu(self.fc4(h)))
        
        if self.split:
            y = self.last[t](h)
            
        else:
            y = self.fc3(h)
        
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
        # self.drop1 = nn.Dropout(0.25)
        # self.drop2 = nn.Dropout(0.5)
        self.drop1 = nn.Dropout(0)
        self.drop2 = nn.Dropout(0)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(int(256*mul),n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, t):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv5(h))
        h=self.relu(self.conv6(h))
#         h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h)))
        
        return self.last[t](h)

class Alexnet(torch.nn.Module):

    def __init__(self,inputsize,taskcla,mul=1):
        super(Alexnet,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        return

    def forward(self, x, t):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        
        return self.last[t](h)


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, inputsize, taskcla, cfg, batch_norm=False):
        super(VGG, self).__init__()

        n_channels, size, _ = inputsize
        self.taskcla = taskcla
        self.layers = make_layers(cfg, n_channels, batch_norm=batch_norm)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            # nn.Dropout(),
            nn.Linear(512*self.smid*self.smid, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Linear(4096, output_dim),
        ])

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(4096,n))
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, t):
        for m in self.layers:
            x = m(x)
        return self.last[t](x)


def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(inputsize,taskcla,mul=1):
    """VGG 11-layer model (configuration "A")"""
    return VGG(inputsize,taskcla, cfg['A'], batch_norm=False)


def vgg11_bn(inputsize,taskcla,mul=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['A'], batch_norm=True)


def vgg13(inputsize,taskcla,mul=1):
    """VGG 13-layer model (configuration "B")"""
    return VGG(inputsize,taskcla, cfg['B'], batch_norm=False)


def vgg13_bn(inputsize,taskcla,mul=1):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['B'], batch_norm=True)


def vgg16(inputsize,taskcla,mul=1):
    """VGG 16-layer model (configuration "D")"""
    return VGG(inputsize,taskcla, cfg['C'], batch_norm=False)


def vgg16_bn(inputsize,taskcla,mul=1):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['C'], batch_norm=True)


def vgg19(inputsize,taskcla,mul=1):
    """VGG 19-layer model (configuration "E")"""
    return VGG(inputsize,taskcla, cfg['D'], batch_norm=False)


def vgg19_bn(inputsize,taskcla,mul=1):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(inputsize,taskcla, cfg['D'], batch_norm=True)

