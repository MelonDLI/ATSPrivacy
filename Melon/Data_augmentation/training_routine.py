import consts

import torch
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt

NON_BLOCKING = False

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
        # Print information about loss and accuracy
        print_status(epoch, loss_fn, optimizer, stats)

        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
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
    #-------------------------------------------------------------------
    # testing validation first before training
    # inputs, targets = next(iter(dataloader)) # single batch
    # batch = 0
    # # for batch, (inputs, targets) in enumerate(dataloader):
    #     # Prep Mini-Batch
    # optimizer.zero_grad()
    #     # Transfer to GPU
    # inputs = inputs.to(**setup)
    # targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
    #     # Get loss
    # outputs = model(inputs)
    # loss, _, _ = loss_fn(outputs, targets)


    # epoch_loss += loss.item()

    # loss.backward()
    # optimizer.step()

    # metric, name, _ = loss_fn.metric(outputs, targets)
    # epoch_metric += metric.item()
    #-------------------------------------------------------------------

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
    path = './plot'
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
    
