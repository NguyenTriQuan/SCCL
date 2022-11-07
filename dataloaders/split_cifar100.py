import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K

def get(args, pc_valid=0.10):
    np.random.seed(args.seed)
    data={}
    taskcla=[]
    size=[3,32,32]
    task_order=shuffle(np.arange(10),random_state=args.seed)
    print('Task order =',task_order+1)

    mean=torch.tensor([x/255 for x in [125.3,123.0,113.9]])
    std=torch.tensor([x/255 for x in [63.0,62.1,66.7]])
    if args.tasknum > 10:
        tasknum = 10
    else:
        tasknum = args.tasknum
    # CIFAR100    
    train_set=datasets.CIFAR100('../dat/',train=True,download=True)
    test_set=datasets.CIFAR100('../dat/',train=False,download=True)

    train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
    test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

    train_data = train_data.permute(0, 3, 1, 2)/255.0
    test_data = test_data.permute(0, 3, 1, 2)/255.0

    n_old = 0
    for t in range(tasknum):
        data[t]={}
        data[t]['name']='cifar100-'+str(task_order[t]+1)
        data[t]['ncla']=10
        #train and valid
        ids = (train_targets//10 == task_order[t])
        images = train_data[ids]
        labels = train_targets[ids]%10 
        if args.cil:
            labels += n_old

        r=np.arange(images.size(0))
        r=np.array(shuffle(r,random_state=args.seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['train_loader'] = DataLoader(TensorDataset(images[itrain], labels[itrain]), batch_size=args.batch_size, shuffle=True)
        data[t]['valid_loader'] = DataLoader(TensorDataset(images[ivalid], labels[ivalid]), batch_size=args.val_batch_size, shuffle=False)

        # data[t]['train_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)

        #test
        ids = (test_targets//10 == task_order[t])
        images = test_data[ids]
        labels = test_targets[ids]%10
        if args.cil:
            labels += n_old
        data[t]['test_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.val_batch_size, shuffle=False)

        n_old += 10

        if args.augment:
            data[t]['train_transform'] = torch.nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), same_on_batch=False),
                K.augmentation.RandomHorizontalFlip(),
                K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=False),
                K.augmentation.RandomGrayscale(p=0.2),
                K.augmentation.Normalize(mean, std),
            )
        else:
            data[t]['train_transform'] = torch.nn.Sequential(
                K.augmentation.Normalize(mean, std),
            )
            
        data[t]['valid_transform'] = torch.nn.Sequential(
            K.augmentation.Normalize(mean, std),
        )
    # Others
    n=0
    for t in range(tasknum):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data, taskcla, size