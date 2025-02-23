"""Implement the .train function."""

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

from collections import defaultdict

from .scheduler import GradualWarmupScheduler

from .. import consts
from ..consts import BENCHMARK, NON_BLOCKING
torch.backends.cudnn.benchmark = BENCHMARK

def train(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None, opt=None):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    print('starting to training model')
    arch = opt.arch
    

    if opt.MoEx:
        print('Mode: MoEx')
    if opt.Mixup:
        print('Mode: Mixup')
        # # MIXUP 
        # print("alpha:{}_{}".format(opt.alpha,opt.alpha*10))
    if not opt.MoEx and not opt.Mixup:
        print('Mode: Regular without MoEx and Mixup')
    print('-----------------------------------------------')
    
    for epoch in range(defs.epochs):
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats, opt,epoch)
        
        # #!save
        # if save_dir is not None:
        #     file = f'{save_dir}/{arch}_{epoch+1}.pth'
        #     # file = f'{save_dir}/model.pth'
        #     torch.save(model.state_dict(), f'{file}')
        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
            model.eval()
            validate(model, loss_fn, validloader, defs, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)
            # if save_dir is not None:
                # file = f'{save_dir}/{arch}_{epoch+1}.pth'
                # file = f'{save_dir}/model.pth'
                # torch.save(model.state_dict(), f'{file}')
        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break

    return stats

criterion = nn.CrossEntropyLoss()

def step(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats, opt,epoch):
    """Step through one epoch."""
    dm = torch.as_tensor(consts.cifar100_mean, **setup)[:, None, None]
    ds = torch.as_tensor(consts.cifar100_std, **setup)[:, None, None]

    epoch_loss, epoch_metric = 0, 0
    inputs, targets = next(iter(dataloader))  #! overfit

    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()
        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
        # print(inputs.shape)
        # print(targets.shape)
        if opt.MoEx and opt.Mixup:
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
            else: # or do not switch MoEx
                outputs = model(inputs)
                loss, _, _ = loss_fn(outputs, targets)
            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, setup, opt.alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss_mixup= lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss = loss +loss_mixup
        elif opt.MoEx:
            # print('MoEx train')
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
            else: # or do not switch MoEx
                outputs = model(inputs)
                loss, _, _ = loss_fn(outputs, targets)
        elif opt.Mixup:
            #Mix up mode
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, setup, opt.alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

        else: # Original ResNet
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)

        # Get loss
        outputs = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)


        epoch_loss += loss.item()

        loss.backward()

        if opt.add_defense:
            if opt.noise_position=='first' and epoch<defs.epochs/4:
                add_defense(opt,model)
            if opt.noise_position=='middle' and epoch>=defs.epochs/4 and epoch<=defs.epochs*3/4:
                add_defense(opt,model)
            if opt.noise_position=='final' and epoch>=defs.epochs*3/4:
                add_defense(opt,model)
            
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
            # if defs.dryrun:
            #     break
        # ! overfit
        batch = 0
        if defs.scheduler == 'linear':
            scheduler.step()

        stats['train_losses'].append(epoch_loss / (batch + 1))
        stats['train_' + name].append(epoch_metric / (batch + 1))
        # stats['train_losses'].append(epoch_loss )
        # stats['train_' + name].append(epoch_metric )

def add_defense(opt,model):
    #! add defense at the second stage of training
    if 'gaussian' in opt.defense:
        if '1e-3' in opt.defense:
            add_noise(model, 1e-3)
        elif '1e-2' in opt.defense:
            add_noise(model, 1e-2)
        else:
            raise NotImplementedError
    elif 'lap' in opt.defense:
        if '1e-3'  in opt.defense:
            lap_noise(model, 1e-3)
        elif '1e-2' in opt.defense:
            lap_noise(model, 1e-2)
        elif '1e-1' in opt.defense:
            lap_noise(model, 1e-1)
        else:
            raise NotImplementedError
        
    elif 'prune' in opt.defense:
        found = False
        for i in [10, 20, 30, 50, 70, 80, 90, 95, 99]:
            if str(i) in opt.defense:
                found=True
                global_prune(model, i)

        if not found:
            raise NotImplementedError

def mixup_data(x, y,setup, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)
    index = index.long()
    index = index.to(**setup)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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


def prune(gradient, percent):
    k = int(gradient.numel() * percent * 0.01)
    shape = gradient.shape
    gradient = gradient.flatten()
    index = torch.topk(torch.abs(gradient), k, largest=False)[1]
    gradient[index] = 0.
    gradient = gradient.view(shape)
    return gradient


def add_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * torch.randn(param.grad.data.shape).cuda()


def lap_sample(shape):
    from torch.distributions.laplace import Laplace
    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    return m.expand(shape).sample()

def lap_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * lap_sample(param.grad.data.shape).cuda()


def global_prune(model, percent):
    for param in model.parameters():
        param.grad.data = prune(param.grad.data, percent)


def step_with_defense(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats, opt):
    """Step through one epoch."""
    assert opt is not None
    dm = torch.as_tensor(consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(consts.cifar10_std, **setup)[:, None, None]

    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()
        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
        # Get loss
        outputs = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)

        epoch_loss += loss.item()
        loss.backward()


        if 'gaussian' in opt.defense:
            if '1e-3' in opt.defense:
                add_noise(model, 1e-3)
            elif '1e-2' in opt.defense:
                add_noise(model, 1e-2)
            else:
                raise NotImplementedError
        elif 'lap' in opt.defense:
            if '1e-3'  in opt.defense:
                lap_noise(model, 1e-3)
            elif '1e-2' in opt.defense:
                lap_noise(model, 1e-2)
            elif '1e-1' in opt.defense:
                lap_noise(model, 1e-1)
            else:
                raise NotImplementedError
        
        elif 'prune' in opt.defense:
            found = False
            for i in [10, 20, 30, 50, 70, 80, 90, 95, 99]:
                if str(i) in opt.defense:
                    found=True
                    global_prune(model, i)

            if not found:
                raise NotImplementedError


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




def train_with_defense(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None, opt=None):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    assert opt is not None
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    print('starting to training model')
    for epoch in range(defs.epochs):
        model.train()
        step_with_defense(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats, opt)

        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
            model.eval()
            validate(model, loss_fn, validloader, defs, setup, stats)
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
