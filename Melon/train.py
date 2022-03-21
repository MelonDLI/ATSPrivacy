from preprocess import preprocess
import utils
import Models
import training_routine
from optimization_strategy import training_strategy
import metrics
import torchvision
import torch
import os
import numpy as np
import argparse

setup = utils.system_startup()

parser = argparse.ArgumentParser(description='Train or evaluate the model by accuracy.')
parser.add_argument('--arch', default='ResNet20-4', type=str, help='Vision model.')
parser.add_argument('--data', default='cifar100', type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help='Vision epoch.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')
# parser.add_argument('--aug_list', default=None, required=True, type=str, help='Augmentation method.')
parser.add_argument('--Moments', default=False,type=bool, help='Switch Moments or not')

opt = parser.parse_args()
arch = opt.arch

def create_save_dir():
    return 'checkpoints/data_{}_arch_{}_rlabel_{}'.format(opt.data, opt.arch, opt.rlabel)


def main():
    defs = training_strategy('conservative'); defs.epochs = opt.epochs #conservative
    # defs = training_strategy('adam'); defs.epochs=opt.epochs # adam
    loss_fn, trainloader, validloader = preprocess(defs=defs)

    # init model
    model = Models.ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=100, num_channels=3, base_width=16 * 4) #RetNet20-4
    model.to(**setup)
    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = f'{save_dir}/{arch}_{defs.epochs}.pth'
    training_routine.train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir)
    torch.save(model.state_dict(), f'{file}')
    model.eval()

def evaluate():
    defs = training_strategy('conservative'); defs.epochs = opt.epochs   
    loss_fn, trainloader, validloader = preprocess(defs=defs)
    model = Models.ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=100, num_channels=3, base_width=16 * 4) #RetNet20-4
    model.to(**setup)

    # reload trained model
    root = create_save_dir()

    filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False
    model.load_state_dict(torch.load(filename))

    model.eval()

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    
    stats = {'valid_losses':list(), 'valid_Accuracy':list()}
    training_routine.validate(model, loss_fn, validloader, defs, setup=setup, stats=stats)
    print(stats)

if __name__ == '__main__':
    print('evaluate: {}'.format(opt.evaluate))
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
