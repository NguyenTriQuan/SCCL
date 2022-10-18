import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader

import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K

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
        data[t]['valid_loader'] = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=False)

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

    return data, taskcla, size

# def get(batch_size, val_batch_size, seed=0, fixed_order=False, pc_valid=0, tasknum = 5):
#     if tasknum>5:
#         tasknum = 5
#     data = {}
#     taskcla = []
#     size = [1, 28, 28]
    
#     # Pre-load
#     # MNIST
#     mean = (0.1307,)
#     std = (0.3081,)
#     if not os.path.isdir('../dat/binary_split_mnist/'):
#         os.makedirs('../dat/binary_split_mnist')
#         dat = {}
#         dat['train'] = datasets.MNIST('../dat/', train=True, download=True, transform=transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
#         dat['test'] = datasets.MNIST('../dat/', train=False, download=True, transform=transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
#         for i in range(5):
#             data[i] = {}
#             data[i]['name'] = 'split_mnist-{:d}'.format(i)
#             data[i]['ncla'] = 2
#             data[i]['train'] = {'x': [], 'y': []}
#             data[i]['test'] = {'x': [], 'y': []}
#         for s in ['train', 'test']:
#             loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
#             for image, target in loader:
#                 task_idx = target.numpy()[0] // 2
#                 data[task_idx][s]['x'].append(image)
#                 data[task_idx][s]['y'].append(target.numpy()[0]%2)

#         for i in range(5):
#             for s in ['train', 'test']:
#                 data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1,size[0],size[1],size[2])
#                 data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
#                 torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin'))
#                 torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
#     else:
#         # Load binary files
#         for i in range(5):
#             data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
#             data[i]['ncla'] = 2
#             data[i]['name'] = 'split_mnist-{:d}'.format(i)

#             # Load
#             for s in ['train', 'test']:
#                 data[i][s] = {'x': [], 'y': []}
#                 data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_mnist'), 'data'+ str(i) + s + 'x.bin')).view(-1,size[0],size[1],size[2])
#                 data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_mnist'), 'data'+ str(i) + s + 'y.bin'))
        
#     for t in range(tasknum):
#         data[t]['valid'] = {}
#         data[t]['valid']['x'] = data[t]['train']['x'].clone()
#         data[t]['valid']['y'] = data[t]['train']['y'].clone()

#         data[t]['train loader'] = DataLoader(
#                     TensorDataset(data[t]['train']['x'], data[t]['train']['y']) , batch_size=batch_size, shuffle=True
#                 )

#         data[t]['valid loader'] = DataLoader(
#                     TensorDataset(data[t]['valid']['x'], data[t]['valid']['y']) , batch_size=val_batch_size, shuffle=False
#                 )

#         data[t]['test loader'] = DataLoader(
#                     TensorDataset(data[t]['test']['x'], data[t]['test']['y']) , batch_size=val_batch_size, shuffle=False
#                 )


#     # Others
#     n = 0
#     for t in range(tasknum):
#         taskcla.append((t, data[t]['ncla']))
#         n += data[t]['ncla']
#     data['ncla'] = n
    
#     return data, taskcla, size