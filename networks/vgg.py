import sys
import torch
import torch.nn as nn
from utils import *

class View(nn.Module):
	"""Changes view using a nn.Module."""

	def __init__(self):
		super(View, self).__init__()

	def forward(self, input):
		return input.view(input.size(0),-1)

class Net(nn.Module):

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
				conv2d = nn.Conv2d(in_channels, int(v * mul), kernel_size=3, padding=1)
				s = compute_conv_output_size(s,3, padding=1)
				if batch_norm:
					bn_layer = nn.ModuleList([nn.BatchNorm2d(int(v * mul)) for _ in range(tasknum)])
					layers += [conv2d, bn_layer, nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
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
		self.last=torch.nn.ModuleList()
		for t,n in self.taskcla:
			self.last.append(nn.Linear(int(4096*mul),n))

	def forward(t, input):
		for module in self.layers:
			if isinstance(module, ModuleList):
				input = module[t](input)
			else:
				input = module(input)

		y = self.last[t](input)
		
		return y
