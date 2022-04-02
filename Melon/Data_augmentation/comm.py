import os, sys
sys.path.insert(0, './')
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from optimization_strategy import training_strategy
from utils import Classification

import policy
policies = policy.policies

from consts import *

def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()

    else:
        raise NotImplementedError
    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0
    # num_workers = 2
    print('num_workers:{}'.format(num_workers))
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
    data_mean, data_std = _get_meanstd(trainset)  # TODO
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

class sub_transform:
    def __init__(self, policy_list):
        self.policy_list = policy_list


    def __call__(self, img):
        idx = np.random.randint(0, len(self.policy_list))
        select_policy = self.policy_list[idx]
        for policy_id in select_policy:
            img = policies[policy_id](img)
        return img


def construct_policy(policy_list):
    if isinstance(policy_list[0], list):
        return sub_transform(policy_list)
    elif isinstance(policy_list[0], int):
        return sub_transform([policy_list])
    else:
        raise NotImplementedError


def build_transform(normalize=True, policy_list=list(), opt=None, defs=None):
    mode = opt.mode
    if opt.data == 'cifar100':
        data_mean, data_std = cifar100_mean, cifar100_std
    else:
        raise NotImplementedError

    # data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if mode !=  'crop':
        transform_list = list()

    elif mode == 'crop':
        transform_list = [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]

    if len(policy_list) > 0 and mode == 'aug':

        transform_list = [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]
        transform_list.append(construct_policy(policy_list))


    # if opt.data == 'FashionMinist':
    #     transform_list = [lambda x: transforms.functional.to_grayscale(x, num_output_channels=3)] + transform_list
    #     transform_list.append(lambda x: transforms.functional.to_grayscale(x, num_output_channels=1))
    #     transform_list.append(transforms.Resize(32))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x),
    ]) 

    print(transform_list)

    transform = transforms.Compose(transform_list)
    return transform


def split(aug_list):
    if '+' not in aug_list:
        return [int(idx) for idx in aug_list.split('-')]
    else:
        ret_list = list()
        for aug in aug_list.split('+'):
            ret_list.append([int(idx) for idx in aug.split('-')])
        return ret_list


def preprocess(opt, defs, valid=False):
    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

        print('preprocess num_workers:{}'.format(num_workers))

    if opt.data == 'cifar100':
        loss_fn, trainloader, validloader =  construct_dataloaders('CIFAR100', defs)
        trainset, validset = _build_cifar100('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []

        if not valid and len(policy_list) > 0:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
                    shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)


        if valid and len(policy_list) > 0:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=defs.batch_size,
                shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

        return loss_fn, trainloader, validloader
    else:
        raise NotImplementedError

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