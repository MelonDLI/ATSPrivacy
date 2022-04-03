import os, sys
sys.path.insert(0, './')
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import torch.nn as nn
import policy

from benchmark.comm import create_model, build_transform, preprocess, create_config



policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Augmentation method.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')

parser.add_argument('--MoEx', default=False,type=bool, help='MoEx or not')
parser.add_argument('--moex_prob', default=0.5,type=float, help='moex_probability')
parser.add_argument('--lam', default=0.9,type=float, help='loss proportion')

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']


def create_save_dir():
    if not opt.MoEx:
        return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    else:
        return 'checkpoints/MoEx_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)



def main():
    if opt.MoEx:
        print("MoEx mode")
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs
    loss_fn, trainloader, validloader = preprocess(opt, defs)

    # init model
    model = create_model(opt)
    model.to(**setup)
    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = f'{save_dir}/{arch}_{defs.epochs}.pth'
    metric_list=inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir, MoEx=opt.MoEx, opt=opt)
    np.save('{}/loss_accuracy_metric.npy'.format(save_dir), metric_list)  #! save accuracy and loss result
    torch.save(model.state_dict(), f'{file}')
    model.eval()


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative'); defs.epochs=opt.epochs
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
    model = create_model(opt)
    model.to(**setup)
    root = create_save_dir()

    filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    stats = {'valid_losses':list(), 'valid_Accuracy':list()}
    inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup=setup, stats=stats)
    print(stats)

if __name__ == '__main__':
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
