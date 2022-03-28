from preprocess import preprocess
import utils
import Models
# import training_routine
from optimization_strategy import training_strategy
import metrics
import torchvision
import torch
import os
import numpy as np
import argparse


import consts
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
NON_BLOCKING = False


setup = utils.system_startup()

parser = argparse.ArgumentParser(description='Train or evaluate the model by accuracy.')
parser.add_argument('--arch', default='ResNet20-4', type=str, help='Vision model.')
parser.add_argument('--data', default='cifar100', type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help='Vision epoch.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')
# parser.add_argument('--aug_list', default=None, required=True, type=str, help='Augmentation method.')
parser.add_argument('--Moments', default=False,type=bool, help='Switch Moments or not')
parser.add_argument('--moex_prob', default=0.5,type=float, help='moex_probability')
parser.add_argument('--lam', default=0.9,type=float, help='loss proportion')

opt = parser.parse_args()
arch = opt.arch

def create_save_dir():
    return 'checkpoints/MoEx_bn_ResNet_20_lam_09_p_05'


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
    train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir)
    torch.save(model.state_dict(), f'{file}')
    model.eval()

def train(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    print('starting to training model')
    for epoch in range(defs.epochs):
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats)
        model.eval()
        validate(model, loss_fn, validloader, defs, setup, stats)
        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
            # model.eval()
            # validate(model, loss_fn, validloader, defs, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)
            if save_dir is not None:
                file = f'{save_dir}/{epoch}.pth'
                torch.save(model.state_dict(), f'{file}')

        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break

    return stats

def step(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats):
    """Step through one epoch."""
    dm = torch.as_tensor(consts.cifar100_mean, **setup)[:, None, None]
    ds = torch.as_tensor(consts.cifar100_std, **setup)[:, None, None]

    epoch_loss, epoch_metric = 0, 0

    input2, target2 = next(iter(dataloader))
    input2 = input2.to(**setup)
    # print('input2 size:{}'.format(input2.shape))
    
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()
        # Transfer to GPU
        inputs = inputs.to(**setup)
        # print('input size:{}'.format(inputs.shape))
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
        
        lam = opt.lam
        r = np.random.rand(1)
        if r < opt.moex_prob: # switch moments
            # generate mixed sample
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            input_a_var = torch.autograd.Variable(inputs, requires_grad=True)
            input_b_var = torch.autograd.Variable(inputs[rand_index], requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)
            outputs = model(input_a_var, input_b_var)
            loss_a, _, _ = loss_fn(outputs, target_a_var)
            loss_b, _, _ = loss_fn(outputs, target_b_var)
            loss = lam*loss_a + (1-lam)*loss_b
        else: # do not switch
            outputs = model(inputs)
        # if input2.shape != inputs.shape:
        #     input2 = input2[:inputs.shape[0],:,:]
        # outputs = model(inputs, input2=input2)  # switch
            # Get loss 
            loss, _, _ = loss_fn(outputs, targets)


        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if defs.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))


def validate(model, loss_fn, dataloader, defs, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, epoch_metric = 0, 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(dataloader):
            # Transfer to GPU
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)

            # Get loss and metric
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            metric, name, _ = loss_fn.metric(outputs, targets)

            epoch_loss += loss.item()
            epoch_metric += metric.item()

            if defs.dryrun:
                break
    stats['valid_losses'].append(epoch_loss / (batch + 1))
    stats['valid_' + name].append(epoch_metric / (batch + 1))


def set_optimizer(model, defs):
    """Build model optimizer and scheduler from defs.
    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
        # Scheduler is fixed to 120 epochs so that calls with fewer epochs are equal in lr drops.

    if defs.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler


def print_status(epoch, loss_fn, optimizer, stats):
    """Print basic console printout every defs.validation epochs."""
    current_lr = optimizer.param_groups[0]['lr']
    name, format = loss_fn.metric()
    print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
          f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{format}} | '
          f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{format}} |')
    save_plot_loss_accuracy(stats,name)


def save_plot_loss_accuracy(stats,name):
    path = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/MoEx_bn_lambda_09/'
    # loss
    fig, ax = plt.subplots(figsize=[8,6])

    tl_line1 = ax.plot(stats['train_losses'], label='Training Loss', color='red')
    vl_line2 = ax.plot(stats['valid_losses'], label='Validation Loss', color='navy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    lines = tl_line1 + vl_line2
    labels = [_.get_label() for _ in lines]
    ax.legend(lines, labels, loc='center right')
    ax.set_xlim(xmin=0)
    fig.savefig('{}/loss.png'.format(path))   # save the figure to file
    plt.close(fig)    # close the figure window

    # accuracy
    fig, ax = plt.subplots(figsize=[8,6])

    tl_line1 = ax.plot(stats["train_" + name], label='Training Accuracy', color='red')
    vl_line2 = ax.plot(stats["valid_" + name], label='Validation Accuracy', color='navy')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')

    lines = tl_line1 + vl_line2
    labels = [_.get_label() for _ in lines]
    ax.legend(lines, labels, loc='center right')
    ax.set_xlim(xmin=0)
    fig.savefig('{}/accuracy.png'.format(path))   # save the figure to file
    plt.close(fig)    # close the figure window

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
