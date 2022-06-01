import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from collections import defaultdict

def get(name, seed=0, pc_valid=0.10):
	data=defaultdict(lambda:{'train': {'x':[], 'y':[]}, 'test': {'x':[], 'y':[]}, 'valid': {'x':[], 'y':[]}})

	if not os.path.isfile('./data/%s.data'%name):
		print('repairing data ...')
		dat = {}
        # CIFAR10
		if name == 'CIFAR10':
			mean=[x/255 for x in [125.3,123.0,113.9]]
			std=[x/255 for x in [63.0,62.1,66.7]]

			dat['train']=datasets.CIFAR10('./data/',train=True,download=True,
	                                      transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
			dat['test']=datasets.CIFAR10('./data/',train=False,download=True,
	                                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        

        # CIFAR100
		elif name == 'CIFAR100':
			mean=[x/255 for x in [125.3,123.0,113.9]]
			std=[x/255 for x in [63.0,62.1,66.7]]
			
			dat['train']=datasets.CIFAR100('./data/',train=True,download=True,
	                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
			dat['test']=datasets.CIFAR100('./data/',train=False,download=True,
	                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
		

		elif name == 'MNIST':
			mean = torch.Tensor([0.1307])
			std = torch.Tensor([0.3081])
	        
			dat['train']=datasets.MNIST('./data/',train=True,download=True,
	                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
			dat['test']=datasets.MNIST('./data/',train=False,download=True,
	                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))


		for s in ['train','test']:
			loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
			for image,target in loader:
				t = target.cpu().numpy()[0]
				data[t][s]['x'].append(image)
				data[t][s]['y'].append(t)

		if 'CIFAR' in name:
			for t in data.keys():
				for s in ['train','test']:
					data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1, 3, 32, 32)
					data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
		elif name == 'MNIST':
			for t in data.keys():
				for s in ['train','test']:
					data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1, 28, 28)
					data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)

		for t in data.keys():
			r=np.arange(data[t]['train']['x'].size(0))
			r=np.array(shuffle(r,random_state=seed),dtype=int)
			nvalid=int(pc_valid*len(r))
			ivalid=torch.LongTensor(r[:nvalid])
			itrain=torch.LongTensor(r[nvalid:])
			data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
			data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
			data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
			data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
			
		data = dict(data)      		
		torch.save(data, './data/%s.data'%name)
    
	# Load binary files
	print('loading data ...')
	data=torch.load('./data/%s.data'%name)

	print('done')
	return data


if __name__ == '__main__':
	get('CIFAR10')

