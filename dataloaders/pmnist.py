import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader


########################################################################################################################
def get(args, pc_valid=0.0):
    np.random.seed(args.seed)
    data={}
    taskcla=[]
    size=[1,28,28]
    task_order=shuffle(np.arange(5),random_state=args.seed)
    print('Task order =',task_order+1)

    mean = (0.1307)
    std = (0.3081)
    tasknum = args.tasknum
    if tasknum > 5:
        tasknum = 5
    # CIFAR100
    dat={}
    
    train_set = datasets.MNIST('../dat/', train=True, download=True)
    test_set = datasets.MNIST('../dat/', train=False, download=True)

    train_data, train_targets = train_set.data.float(), train_set.targets.long()
    test_data, test_targets = test_set.data.float(), test_set.targets.long()

    train_data = train_data.unsqueeze(1)/255.0
    test_data = test_data.unsqueeze(1)/255.0

    # train_data = (train_data - mean.view(1,-1,1,1))/std.view(1,-1,1,1)
    # test_data = (test_data - mean.view(1,-1,1,1))/std.view(1,-1,1,1)
    n_old = 0
    for t in range(tasknum):
        print(t, end=',')
        sys.stdout.flush()
        data[t] = {}
        data[t]['name'] = 'pmnist-{:d}'.format(i)
        data[t]['ncla'] = 10
        permutation = np.random.permutation(28*28)

        data[t]={}
        data[t]['name']='mnist-'+str(task_order[t]+1)
        data[t]['ncla']=2
        #train and valid
        ids = (train_targets//2 == task_order[t])
        images = train_data[ids]
        labels = train_targets[ids]%2
        if args.cil:
            labels += n_old

        data[t]['train_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=True)
        data[t]['valid_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.val_batch_size, shuffle=False)

        #test
        ids = (test_targets//2 == task_order[t])
        images = test_data[ids]
        labels = test_targets[ids]%2
        if args.cil:
            labels += n_old
        data[t]['test_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.val_batch_size, shuffle=False)

        n_old += 2

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


def get(batch_size, val_batch_size, seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    np.random.seed(seed)
    data = {}
    taskcla = []
    size = [1, 28, 28]
    # Pre-load
    # MNIST
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    dat = {}
    dat['train'] = datasets.MNIST('../dat/', train=True, download=True)
    dat['test'] = datasets.MNIST('../dat/', train=False, download=True)
    
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'pmnist-{:d}'.format(i)
        data[i]['ncla'] = 10
        permutation = np.random.permutation(28*28)
        for s in ['train', 'test']:
            if s == 'train':
                arr = dat[s].train_data.view(dat[s].train_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].train_labels)
            else:
                arr = dat[s].test_data.view(dat[s].test_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].test_labels)
                
            arr = (arr/255 - mean) / std
            data[i][s]={}
            data[i][s]['x'] = arr[:,permutation].view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = label
            
    # Validation
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

        data[t]['train loader'] = DataLoader(
                    TensorDataset(data[t]['train']['x'], data[t]['train']['y']) , batch_size=batch_size, shuffle=True
                )

        data[t]['valid loader'] = DataLoader(
                    TensorDataset(data[t]['valid']['x'], data[t]['valid']['y']) , batch_size=val_batch_size, shuffle=False
                )

        data[t]['test loader'] = DataLoader(
                    TensorDataset(data[t]['test']['x'], data[t]['test']['y']) , batch_size=val_batch_size, shuffle=False
                )
        
        # r=np.arange(data[t]['train']['x'].size(0))
        # r=np.array(shuffle(r,random_state=seed),dtype=int)
        # nvalid=int(pc_valid*len(r))
        # ivalid=torch.LongTensor(r[:10000])
        # itrain=torch.LongTensor(r[10000:])
        # data[t]['valid']={}
        # data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        # data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        # data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        # data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

########################################################################################################################