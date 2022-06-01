import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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

inputsize = [3, 32, 32]
taskcla = [(i, 5) for i in range(20)]

def cifar100_train_loader(dataset_name, train_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/content/cifar100_org/train/{}'.format(dataset_name)
    # data_dir = '../dat/cifar100_org/train/{}'.format(dataset_name)

    train_dataset = datasets.ImageFolder(data_dir, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None, pin_memory=pin_memory)


def cifar100_prune_loader(dataset_name, train_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = '/content/cifar100_org/train/{}'.format(dataset_name)
    # data_dir = '../dat/cifar100_org/train/{}'.format(dataset_name)

    train_dataset = datasets.ImageFolder(data_dir, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None, pin_memory=pin_memory)


def cifar100_val_loader(dataset_name, val_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    data_dir = '/content/cifar100_org/test/{}'.format(dataset_name)
    # data_dir = '../dat/cifar100_org/test/{}'.format(dataset_name)
    val_dataset = \
        datasets.ImageFolder(data_dir,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None, pin_memory=pin_memory)
