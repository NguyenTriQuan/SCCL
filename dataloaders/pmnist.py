import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K

########################################################################################################################
def get(args, pc_valid=0.0):
    np.random.seed(args.seed)
    data={}
    taskcla=[]
    size=[1,28,28]
    # task_order=shuffle(np.arange(5),random_state=args.seed)
    # print('Task order =',task_order+1)

    mean = (0.1307)
    std = (0.3081)
    tasknum = args.tasknum
    # CIFAR100
    dat={}
    
    train_set = datasets.MNIST('../dat/', train=True, download=True)
    test_set = datasets.MNIST('../dat/', train=False, download=True)

    train_data, train_targets = train_set.data.float(), train_set.targets.long()
    test_data, test_targets = test_set.data.float(), test_set.targets.long()

    train_data = train_data.unsqueeze(1)/255.0
    test_data = test_data.unsqueeze(1)/255.0

    # train_data = (train_data-mean)/std
    # test_data = (test_data-mean)/std

    n_old = 0
    for t in range(tasknum):
        print(t, end=',')
        sys.stdout.flush()
        data[t] = {}
        data[t]['name'] = 'pmnist-{:d}'.format(t)
        data[t]['ncla'] = 10
        permutation = torch.randperm(28*28)

        if args.cil:
            train_targets += n_old

        images = train_data.view(train_data.shape[0], -1)[:, permutation].view(train_data.shape[0], 1, 28, 28)
        data[t]['train_loader'] = DataLoader(TensorDataset(images, train_targets), batch_size=args.batch_size, shuffle=True)
        data[t]['valid_loader'] = DataLoader(TensorDataset(images, train_targets), batch_size=args.val_batch_size, shuffle=False)

        #test
        images = test_data.view(test_data.shape[0], -1)[:, permutation].view(test_data.shape[0], 1, 28, 28)
        if args.cil:
            train_targets += n_old

        data[t]['test_loader'] = DataLoader(TensorDataset(images, test_targets), batch_size=args.val_batch_size, shuffle=False)

        n_old += 10

    if args.augment:
        data['train_transform'] = torch.nn.Sequential(
            K.augmentation.RandomResizedCrop(size=(28, 28), scale=(0.2, 1.0), same_on_batch=False),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.Normalize(mean, std),
        )
    else:
        data['train_transform'] = torch.nn.Sequential(
            K.augmentation.Normalize(mean, std),
        )
        
    data['valid_transform'] = torch.nn.Sequential(
        K.augmentation.Normalize(mean, std),
    )
    # Others
    n=0
    for t in range(tasknum):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    print()
    return data, taskcla, size

########################################################################################################################