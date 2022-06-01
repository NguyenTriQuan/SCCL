import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import  TensorDataset, DataLoader
from .cifar100_config import *

DATASETS=[
    'aquatic_mammals',
    'fish',
    'flowers',
    'food_containers',
    'fruit_and_vegetables',
    'household_electrical_devices',
    'household_furniture',
    'insects',
    'large_carnivores',
    'large_man-made_outdoor_things',
    'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores',
    'medium_mammals',
    'non-insect_invertebrates',
    'people',
    'reptiles',
    'small_mammals',
    'trees',
    'vehicles_1',
    'vehicles_2',
]


def get(batch_size, val_batch_size, seed=0, tasknum=1):

    inputsize = [3, 32, 32]
    taskcla = []
    data = {}
    for i, dataset_name in enumerate(DATASETS):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        data[i]['ncla'] = 5
        data[i]['name'] = dataset_name

        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


        train_dir = '/content/cifar100_org/train/{}'.format(dataset_name)
        test_dir = '/content/cifar100_org/test/{}'.format(dataset_name)

        train_dataset = datasets.ImageFolder(train_dir, train_transform)
        val_dataset = datasets.ImageFolder(test_dir, 
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

        data[t]['train loader'] = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

        data[t]['valid loader'] = DataLoader(
                    val_dataset, batch_size=val_batch_size, shuffle=False
                )

        data[t]['test loader'] = DataLoader(
                    val_dataset, batch_size=val_batch_size, shuffle=False
                )

    n = 0
    for t in range(20):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, inputsize

