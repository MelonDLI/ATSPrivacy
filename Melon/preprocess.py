import os, sys
sys.path.insert(0, './')
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from optimization_strategy import training_strategy
from utils import Classification
# import argparse

# opt = parser.parse_args()

def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()

    # if MULTITHREAD_DATAPROCESSING:
    #     num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    # else:
    #     num_workers = 0
    num_workers = 2

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers,pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(validset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers,pin_memory=True)

    return loss_fn, trainloader, validloader

def _get_meanstd(trainset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    # if cifar100_mean is None:
    data_mean, data_std = _get_meanstd(trainset)
    # else:
    #     data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def preprocess(opt=None, defs=None, valid=False):
    # if opt.data == 'cifar100':
    loss_fn, trainloader, validloader =  construct_dataloaders('CIFAR100', defs)
    trainset, validset = _build_cifar100('~/data/')

    # if len(opt.aug_list) > 0:
    #     policy_list = split(opt.aug_list)
    # else:
        #     policy_list = []
    policy_list = []
    # if not valid:
    #     trainset.transform = build_transform(True, policy_list, opt, defs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
                shuffle=True, drop_last=False, num_workers=2, pin_memory=True)


    # if valid:
    #     validset.transform = build_transform(True, policy_list, opt, defs)
    validloader = torch.utils.data.DataLoader(validset, batch_size=defs.batch_size,
            shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    return loss_fn, trainloader, validloader

    # else:
    #     raise NotImplementedError


# if __name__ == '__main__':
#     defs = training_strategy('conservative'); defs.epochs = 100
#     loss_fn, trainloader, validloader = preprocess(defs=defs)
#     # print(loss_fn)
#     # print(trainloader)
#     data, targets = next(iter(trainloader))
#     print(targets) # batch size 128

def create_config(opt):
    # TODO opt
    # Tried Method 1
    if opt.optim == 'inversed':
        config = dict(signed=True,
                    boxed=True,
                    cost_fn='sim',
                    indices='def',
                    weights='equal',
                    lr=0.1,
                    optim='adam',
                    restarts=1,
                    max_iterations=4800,
                    total_variation=1e-4,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')

    return config