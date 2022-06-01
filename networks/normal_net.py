import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Bernoulli, LogNormal, Normal

class MLP(nn.Module):
	"""docstring for Net"""
	def __init__(self, N):
		super(MLP, self).__init__()
		N1, N2 = N
		self.fc1 = nn.Linear(28*28, N1)
		# nn.init.uniform_(self.fc1.bias, 0, 0)
		# nn.init.xavier_uniform_(self.fc1.weight)
		self.fc2 = nn.Linear(N1, N2)
		# nn.init.uniform_(self.fc2.bias, 0, 0)
		# nn.init.xavier_uniform_(self.fc2.weight)
		self.fc3 = nn.Linear(N2, 10)
		self.activate = nn.ReLU()
		self.name = 'mlp'
		
		# self.mask0 = nn.Parameter(torch.ones(28*28).bool().cuda(), requires_grad=False)
		self.mask1 = nn.Parameter(torch.ones(N1).bool().cuda(), requires_grad=False)
		self.mask2 = nn.Parameter(torch.ones(N2).bool().cuda(), requires_grad=False)
		self.dropout1= nn.Dropout(0.2)
		self.dropout2= nn.Dropout(0.5)
		self.drop1 = DropoutLayer(N1, self.fc1.weight, self.fc2.weight)
		self.drop2 = DropoutLayer(N2, self.fc2.weight, self.fc3.weight)
		self.masks = [self.mask1, self.mask2]


	def forward(self, x):
		x = x.view(x.size(0),-1)
		x = self.dropout1(x)

		h1 = self.fc1(x)
		x = self.activate(h1)
		# x = self.dropout2(x)
		# x = self.drop1(x)
		x = x*self.mask1

		h2 = self.fc2(x)
		x = self.activate(h2)
		# x = self.dropout2(x)
		# x = self.drop2(x)
		x = x*self.mask2

		h3 = self.fc3(x)
		return h1, h2, h3


	def group_fc(self, layer, dim):
		return layer.weight.norm(2, dim=dim)

	def update_mask(self, thres):
		# self.mask1.data *= self.drop1.get_mask()
		# self.mask2.data *= self.drop2.get_mask()
		self.mask1.data *= (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres)
		self.mask2.data *= (self.group_fc(self.fc3, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)

	def get_mask(self, thres):
		mask1 = (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres)
		mask2 = (self.group_fc(self.fc3, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)
		return [mask1, mask2]

	def group_lasso_reg(self, lamb1, lamb2):
		reg = 0
		reg += lamb1*(self.group_fc(self.fc1, 1).sum() + self.group_fc(self.fc2, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2, 1).sum() + self.group_fc(self.fc3, 0).sum())
		return reg

	def kl_divergence(self):
		return self.drop1.kl_divergence() + self.drop2.kl_divergence()

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
	return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.name = 'cnn'
		self.c1 = nn.Conv2d(3,64,kernel_size=3,padding=1)
		s = compute_conv_output_size(32,3, padding=1) # 32
		s = s//2 # 16
		self.c3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
		s = compute_conv_output_size(s,3, padding=1) # 16
		s = s//2 # 8
		self.c5 = nn.Conv2d(128,256,kernel_size=3,padding=1)
		s = compute_conv_output_size(s,3, padding=1) # 8

		s = s//2 # 4
		self.fc1 = nn.Linear(s*s*256,2048) # 2048
		self.fc2 = nn.Linear(2048,2048)
		self.drop1=torch.nn.Dropout(0.2)
		self.drop2=torch.nn.Dropout(0.5)
		self.smid=s
		self.MaxPool = torch.nn.MaxPool2d(2)
		self.relu=torch.nn.ReLU()

		self.last=torch.nn.Linear(2048,10)

		self.mask1 = nn.Parameter(torch.ones(64).bool().cuda(), requires_grad=False)
		self.mask3 = nn.Parameter(torch.ones(128).bool().cuda(), requires_grad=False)
		self.mask5 = nn.Parameter(torch.ones(256).bool().cuda(), requires_grad=False)
		self.mask6 = nn.Parameter(torch.ones(2048).bool().cuda(), requires_grad=False)
		self.mask8 = nn.Parameter(torch.ones(2048).bool().cuda(), requires_grad=False)

		self.masks = [self.mask1, self.mask3, self.mask5, self.mask6, self.mask8]


	def forward(self, x):
		h=self.relu(self.c1(x))
		# h=h*self.mask1.view(1,-1,1,1).expand_as(h)
		h=self.MaxPool(h)
        
		h=self.relu(self.c3(h))
		# h=h*self.mask3.view(1,-1,1,1).expand_as(h)
		h=self.MaxPool(h)
		
		h=self.relu(self.c5(h))
		# h=h*self.mask5.view(1,-1,1,1).expand_as(h)
		h=self.MaxPool(h)
        
		h=h.view(x.shape[0],-1)
		h = self.drop2(h)

		h=self.relu(self.fc1(h))
		# h=h*self.mask6

		h=self.relu(self.fc2(h))
		# h=h*self.mask8

		h = self.last(h)
		return h

	def group_conv(self, layer, dim):
		return layer.weight.norm(2, dim=(dim, 2, 3))

	def group_fc(self, layer, dim):
		return layer.weight.norm(2, dim=dim)

	def get_mask(self, thres):
		
		mask1 = (self.group_conv(self.c3, 0)>thres)*(self.group_conv(self.c1, 1)>thres)
		mask3 = (self.group_conv(self.c5, 0)>thres)*(self.group_conv(self.c3, 1)>thres)
		mask5 = (self.group_conv(self.c5, 1)>thres)
		mask6 = (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres)
		mask8 = (self.group_fc(self.last, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)
		return [mask1, mask3, mask5, mask6, mask8]

	def update_mask(self, thres):
		
		self.mask1.data *= (self.group_conv(self.c3, 0)>thres)*(self.group_conv(self.c1, 1)>thres)
		self.mask3.data *= (self.group_conv(self.c5, 0)>thres)*(self.group_conv(self.c3, 1)>thres)
		self.mask5.data *= (self.group_conv(self.c5, 1)>thres)
		self.mask6.data *= (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres)
		self.mask8.data *= (self.group_fc(self.last, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)

	def group_lasso_reg(self, lamb1, lamb2):
		reg = 0
		reg += lamb1*(self.group_conv(self.c1, 1).sum() + self.group_conv(self.c3, 0).sum())
		reg += lamb1*(self.group_conv(self.c3, 1).sum() + self.group_conv(self.c5, 0).sum())
		reg += lamb1*self.group_conv(self.c5, 1).sum()
		reg += lamb2*(self.group_fc(self.fc1, 1).sum() + self.group_fc(self.fc2, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2, 1).sum() + self.group_fc(self.last, 0).sum())
		return reg


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Sequential_Debug(nn.Sequential):
	def forward(self, input):
		for module in self._modules.values():
			input = module(input)
		return input

class View(nn.Module):
	"""Changes view using a nn.Module."""

	def __init__(self, *shape):
		super(View, self).__init__()
		self.shape = shape

	def forward(self, input):
		return input.view(*self.shape)

class VGG(nn.Module):

	def __init__(self, cfg, batch_norm=True):
		super(VGG, self).__init__()
		self.name = 'vgg'
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		layers += [
			View(-1, 512),
			nn.Linear(512, 4096),
			nn.ReLU(True),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Linear(4096, 5)
		]
		self.layers = Sequential_Debug(*layers)
		self._initialize_weights()

	def forward(self, x, t=0):
		return self.layers(x)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
		

class DropoutLayer(nn.Module):
	"""docstring for ClassName"""
	def __init__(self, in_features, prev_weight, post_weight):
		super(DropoutLayer, self).__init__()
		self.in_features = in_features
		self.mu = nn.Parameter(torch.Tensor(in_features).uniform_(1.0, 1.0))
		self.log_sigma2 = nn.Parameter(torch.Tensor(in_features).uniform_(0.0, 0.0))
		self.prev_weight = prev_weight
		self.post_weight = post_weight
		self.normal = Normal(0, 1)
		self.thres = 1e-2

	def forward(self, x): 
		if self.training:
			sigma = torch.exp(0.5*self.log_sigma2)
			epsilon = self.normal.sample(self.log_sigma2.size()).cuda()
			x = x * (self.mu + sigma * epsilon)
		else:
			x = x * self.mu
		# x = x * self.mu
		return x

	def kl_divergence(self):
		kld = 0.5*torch.log1p(self.mu*self.mu/(torch.exp(self.log_sigma2)+1e-8))
		return kld.sum()

	@property
	def log_alpha(self):
		return self.log_sigma2 - 2.0 * torch.log(self.mu + 1e-8)
	# @property
	# def mu(self):
	# 	return (self.prev_weight.norm(2, dim=1)>self.thres) * (self.post_weight.norm(2, dim=0)>self.thres)

	def get_mask(self):
		alpha = torch.exp(self.log_alpha)
		p = alpha / (1+alpha)
		return (p < 0.5)


class DynamicMLP(nn.Module):
	"""docstring for Net"""
	def __init__(self, N):
		super(DynamicMLP, self).__init__()
		N1, N2, N3 = N
		self.fc1 = nn.Linear(28*28, N1)
		self.fc2 = nn.Linear(N1, N2)
		self.fc3 = nn.Linear(N2, N3)
		self.activate = nn.ReLU()
		self.name = 'mlp'
		self.drop1=torch.nn.Dropout(0.2)
		self.drop2=torch.nn.Dropout(0.1)

	def forward(self, x, dropout=-1):
		x = x.view(x.size(0),-1)
		if dropout == 0:
			x = self.drop1(x)
		x = self.fc1(x)
		x = self.activate(x)
		if dropout == 1:
			x = self.drop1(x)

		x = self.fc2(x)
		x = self.activate(x)
		if dropout == 2:
			x = self.drop1(x)

		x = self.fc3(x)
		return x


	def group_fc(self, weight, dim):
		return weight.norm(2, dim=dim) * math.sqrt(weight.shape[dim])

	def mask_fc(self, weight, dim, thres):
		# shape = weight.shape
		# neurons_shape = [shape[dim]]
		return weight.norm(2, dim=dim) > thres #*math.sqrt(np.prod(neurons_shape))

	def squeeze(self, thres):
		mask_out = self.mask_fc(self.fc2.weight, 0, thres)*self.mask_fc(self.fc1.weight, 1, thres) 
		self.fc1.weight = nn.Parameter(self.fc1.weight.data[mask_out].clone())
		self.fc1.bias = nn.Parameter(self.fc1.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = self.mask_fc(self.fc3.weight, 0, thres)*self.mask_fc(self.fc2.weight, 1, thres)
		self.fc2.weight = nn.Parameter(self.fc2.weight.data[mask_out][:,mask_in].clone())
		self.fc2.bias = nn.Parameter(self.fc2.bias.data[mask_out].clone())

		mask_in = mask_out
		self.fc3.weight = nn.Parameter(self.fc3.weight.data[:,mask_in].clone())

	def group_lasso_reg(self, lamb1, lamb2):
		reg = 0
		reg += lamb1*(self.group_fc(self.fc1.weight, 1).sum() + self.group_fc(self.fc2.weight, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2.weight, 1).sum() + self.group_fc(self.fc3.weight, 0).sum())
		return reg

	def sparse_group_lasso_reg(self, lamb1, lamb2, alpha=0.5):
		reg = 0
		reg += lamb1*(self.group_fc(self.fc1.weight, 1).sum() + self.group_fc(self.fc2.weight, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2.weight, 1).sum() + self.group_fc(self.fc3.weight, 0).sum())
		reg += lamb1*self.fc1.weight.norm(1)
		reg += lamb1*self.fc2.weight.norm(1)
		reg += lamb2*self.fc3.weight.norm(1)
		return reg



class DynamicCNN(nn.Module):
	def __init__(self, N):
		super(DynamicCNN, self).__init__()
		self.name = 'cnn'
		nc1, nc3, nc5, nfc1, nfc2, nfc3 = N
		self.c1 = nn.Conv2d(3,nc1,kernel_size=3,padding=1)
		s = compute_conv_output_size(32,3, padding=1) # 32
		s = s//2 # 16
		self.c3 = nn.Conv2d(nc1,nc3,kernel_size=3,padding=1)
		s = compute_conv_output_size(s,3, padding=1) # 16
		s = s//2 # 8
		self.c5 = nn.Conv2d(nc3,nc5,kernel_size=3,padding=1)
		s = compute_conv_output_size(s,3, padding=1) # 8

		s = s//2 # 4
		self.fc1 = nn.Linear(s*s*nc5,nfc1) # 2048
		self.fc2 = nn.Linear(nfc1,nfc2)
		self.drop1=torch.nn.Dropout(0.2)
		self.drop2=torch.nn.Dropout(0.5)
		self.smid=s
		self.MaxPool = torch.nn.MaxPool2d(2)
		self.relu=torch.nn.ReLU()

		self.last=torch.nn.Linear(nfc2,nfc3)


	def forward(self, x, dropout=False):
		# x=self.drop1(x)
		h=self.relu(self.c1(x))
		# x=self.drop1(x)
		h=self.MaxPool(h)
        
		h=self.relu(self.c3(h))
		# x=self.drop1(x)
		h=self.MaxPool(h)
		
		h=self.relu(self.c5(h))
		# x=self.drop1(x)
		h=self.MaxPool(h)
        
		h=h.view(x.shape[0],-1)
		if dropout:
			h=self.drop2(h)

		h=self.relu(self.fc1(h))

		h=self.relu(self.fc2(h))

		h = self.last(h)
		return h

	def group_conv(self, weight, dim):
		return weight.norm(2, dim=(dim, 2, 3))

	def group_fc(self, weight, dim):
		return weight.norm(2, dim=dim)

	def mask_conv(self, weight, dim, thres):
		# shape = weight.shape
		# neurons_shape = [shape[dim], shape[2], shape[3]]
		return weight.norm(2, dim=(dim, 2, 3)) > thres #*math.sqrt(np.prod(neurons_shape))

	def mask_fc(self, weight, dim, thres):
		# shape = weight.shape
		# neurons_shape = [shape[dim]]
		return weight.norm(2, dim=dim) > thres #*math.sqrt(np.prod(neurons_shape))

	def squeeze(self, thres):

		mask_out = self.mask_conv(self.c3.weight, 0, thres)*self.mask_conv(self.c1.weight, 1, thres)
		self.c1.weight = nn.Parameter(self.c1.weight.data[mask_out].clone())
		self.c1.bias = nn.Parameter(self.c1.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = self.mask_conv(self.c5.weight, 0, thres)*self.mask_conv(self.c3.weight, 1, thres)
		self.c3.weight = nn.Parameter(self.c3.weight.data[mask_out][:,mask_in].clone())
		self.c3.bias = nn.Parameter(self.c3.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = self.mask_conv(self.c5.weight, 1, thres)*self.mask_conv(
			self.fc1.weight.view(self.fc1.weight.shape[0], self.c5.weight.shape[0], self.smid, self.smid), 0, thres)
		self.c5.weight = nn.Parameter(self.c5.weight.data[mask_out][:,mask_in].clone())
		self.c5.bias = nn.Parameter(self.c5.bias.data[mask_out].clone())

		mask_in = mask_out.view(-1,1,1).expand(mask_out.size(0),self.smid,self.smid).contiguous().view(-1)
		mask_out = self.mask_fc(self.fc2.weight, 0, thres)*self.mask_fc(self.fc1.weight, 1, thres) 
		self.fc1.weight = nn.Parameter(self.fc1.weight.data[mask_out][:,mask_in].clone())
		self.fc1.bias = nn.Parameter(self.fc1.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = self.mask_fc(self.last.weight, 0, thres)*self.mask_fc(self.fc2.weight, 1, thres)
		self.fc2.weight = nn.Parameter(self.fc2.weight.data[mask_out][:,mask_in].clone())
		self.fc2.bias = nn.Parameter(self.fc2.bias.data[mask_out].clone())

		mask_in = mask_out
		self.last.weight = nn.Parameter(self.last.weight.data[:,mask_in].clone())


	def group_lasso_reg(self, lamb1, lamb2):
		reg = 0
		reg += lamb1*(self.group_conv(self.c1.weight, 1).sum() + self.group_conv(self.c3.weight, 0).sum())
		reg += lamb1*(self.group_conv(self.c3.weight, 1).sum() + self.group_conv(self.c5.weight, 0).sum())
		reg += lamb1*(self.group_conv(self.c5.weight, 1).sum() + 
			self.group_conv(self.fc1.weight.view(self.fc1.weight.shape[0], self.c5.weight.shape[0], self.smid, self.smid), 0).sum())
		reg += lamb1*(self.group_fc(self.fc1.weight, 1).sum() + self.group_fc(self.fc2.weight, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2.weight, 1).sum() + self.group_fc(self.last.weight, 0).sum())

		# reg += lamb1*self.c1.weight.norm(1)
		# reg += lamb1*self.c3.weight.norm(1)
		# reg += lamb1*self.c5.weight.norm(1)
		# reg += lamb2*self.fc1.weight.norm(1)
		# reg += lamb2*self.fc2.weight.norm(1)
		# reg += lamb2*self.last.weight.norm(1)

		return reg

	def sparse_group_lasso_reg(self, lamb1, lamb2, alpha=0.5):
		reg = 0
		reg += lamb1*(self.group_conv(self.c1.weight, 1).sum() + self.group_conv(self.c3.weight, 0).sum())
		reg += lamb1*(self.group_conv(self.c3.weight, 1).sum() + self.group_conv(self.c5.weight, 0).sum())
		reg += lamb1*(self.group_conv(self.c5.weight, 1).sum() + 
			self.group_conv(self.fc1.weight.view(self.fc1.weight.shape[0], self.c5.weight.shape[0], self.smid, self.smid), 0))
		reg += lamb1*(self.group_fc(self.fc1.weight, 1).sum() + self.group_fc(self.fc2.weight, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2.weight, 1).sum() + self.group_fc(self.last.weight, 0).sum())

		reg += lamb1*self.c1.weight.norm(1)
		reg += lamb1*self.c3.weight.norm(1)
		reg += lamb1*self.c5.weight.norm(1)
		reg += lamb2*self.fc1.weight.norm(1)
		reg += lamb2*self.fc2.weight.norm(1)
		reg += lamb2*self.last.weight.norm(1)

		return reg


class DynamicCNN_MNIST(nn.Module):
	def __init__(self):
		super(DynamicCNN_MNIST, self).__init__()
		self.name = 'cnn'
		self.c1 = nn.Conv2d(1,20,kernel_size=3,padding=1)
		s = compute_conv_output_size(28,3, padding=1) # 32
		s = s//2 # 16
		self.c3 = nn.Conv2d(20,50,kernel_size=3,padding=1)
		s = compute_conv_output_size(s,3, padding=1) # 16
		s = s//2 # 8
		# self.c5 = nn.Conv2d(128,256,kernel_size=3,padding=1)
		# s = compute_conv_output_size(s,3, padding=1) # 8

		# s = s//2 # 4
		self.fc1 = nn.Linear(s*s*50,800) # 2048
		self.fc2 = nn.Linear(800,500)
		self.drop1=torch.nn.Dropout(0.2)
		self.drop2=torch.nn.Dropout(0.5)
		self.smid=s
		self.MaxPool = torch.nn.MaxPool2d(2)
		self.relu=torch.nn.ReLU()

		self.last=torch.nn.Linear(500,10)


	def forward(self, x, dropout=False):
		# x=self.drop1(x)
		h=self.relu(self.c1(x))
		# x=self.drop1(x)
		h=self.MaxPool(h)
        
		h=self.relu(self.c3(h))
		# x=self.drop1(x)
		h=self.MaxPool(h)
		
		# h=self.relu(self.c5(h))
		# # x=self.drop1(x)
		# h=self.MaxPool(h)
        
		h=h.view(x.shape[0],-1)
		if dropout:
			h=self.drop2(h)

		h=self.relu(self.fc1(h))

		h=self.relu(self.fc2(h))

		h = self.last(h)
		return h

	def group_conv(self, layer, dim):
		return layer.weight.norm(2, dim=(dim, 2, 3))

	def group_fc(self, layer, dim):
		return layer.weight.norm(2, dim=dim)

	def squeeze(self, thres):

		mask_out = (self.group_conv(self.c3, 0)>thres)*(self.group_conv(self.c1, 1)>thres)
		self.c1.weight = nn.Parameter(self.c1.weight.data[mask_out].clone())
		self.c1.bias = nn.Parameter(self.c1.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = (self.group_conv(self.c3, 1)>thres)
		self.c3.weight = nn.Parameter(self.c3.weight.data[mask_out][:,mask_in].clone())
		self.c3.bias = nn.Parameter(self.c3.bias.data[mask_out].clone())

		mask_in = mask_out.view(-1,1,1).expand(mask_out.size(0),self.smid,self.smid).contiguous().view(-1)
		mask_out = (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres) 
		self.fc1.weight = nn.Parameter(self.fc1.weight.data[mask_out][:,mask_in].clone())
		self.fc1.bias = nn.Parameter(self.fc1.bias.data[mask_out].clone())

		mask_in = mask_out
		mask_out = (self.group_fc(self.last, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)
		self.fc2.weight = nn.Parameter(self.fc2.weight.data[mask_out][:,mask_in].clone())
		self.fc2.bias = nn.Parameter(self.fc2.bias.data[mask_out].clone())

		mask_in = mask_out
		self.last.weight = nn.Parameter(self.last.weight.data[:,mask_in].clone())

	def update_mask(self, thres):
		
		self.mask1.data *= (self.group_conv(self.c3, 0)>thres)*(self.group_conv(self.c1, 1)>thres)
		self.mask3.data *= (self.group_conv(self.c3, 1)>thres)
		self.mask6.data *= (self.group_fc(self.fc2, 0)>thres)*(self.group_fc(self.fc1, 1)>thres)
		self.mask8.data *= (self.group_fc(self.last, 0)>thres)*(self.group_fc(self.fc2, 1)>thres)

	def group_lasso_reg(self, lamb1, lamb2):
		reg = 0
		reg += lamb1*(self.group_conv(self.c1, 1).sum() + self.group_conv(self.c3, 0).sum())
		reg += lamb1*self.group_conv(self.c3, 1).sum()
		reg += lamb2*(self.group_fc(self.fc1, 1).sum() + self.group_fc(self.fc2, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2, 1).sum() + self.group_fc(self.last, 0).sum())

		# reg += lamb1*self.c1.weight.norm(1)
		# reg += lamb1*self.c3.weight.norm(1)
		# reg += lamb1*self.c5.weight.norm(1)
		# reg += lamb2*self.fc1.weight.norm(1)
		# reg += lamb2*self.fc2.weight.norm(1)
		# reg += lamb2*self.last.weight.norm(1)

		return reg

	def sparse_group_lasso_reg(self, lamb1, lamb2, alpha=0.5):
		reg = 0
		reg += lamb1*(self.group_conv(self.c1, 1).sum() + self.group_conv(self.c3, 0).sum())
		reg += lamb1*(self.group_conv(self.c3, 1).sum() + self.group_conv(self.c5, 0).sum())
		reg += lamb1*self.group_conv(self.c5, 1).sum()
		reg += lamb2*(self.group_fc(self.fc1, 1).sum() + self.group_fc(self.fc2, 0).sum())
		reg += lamb2*(self.group_fc(self.fc2, 1).sum() + self.group_fc(self.last, 0).sum())

		reg += lamb1*self.c1.weight.norm(1)
		reg += lamb1*self.c3.weight.norm(1)
		reg += lamb1*self.c5.weight.norm(1)
		reg += lamb2*self.fc1.weight.norm(1)
		reg += lamb2*self.fc2.weight.norm(1)
		reg += lamb2*self.last.weight.norm(1)

		return reg