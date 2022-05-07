import os, sys
sys.path.insert(0, './')
import inversefed
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
import policy
from benchmark.comm import create_model, build_transform, preprocess, create_config


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')

#MoEx
parser.add_argument('--MoEx', default =False, type=bool,help='MoEx or not')
#Mixup
parser.add_argument('--Mixup',default=False, type=bool,help='Mix up or not')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
# # defense
# parser.add_argument('--add_defense',default=False,type=bool,help='add defense or not')
# parser.add_argument('--defense', action='store',
#                     type=str, nargs='*', default=['prune','95'],
#                     help="defense type")
# parser.add_argument('--noise_position',default='middle',type=str,help='where add noise to model training')

opt = parser.parse_args()
num_images = 1


# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs


# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']

config = create_config(opt)


def create_save_dir():
    return 'benchmark_copy/corrupt_label_image'


def reconstruct(idx, model, loss_fn, trainloader, validloader):

    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)


    # attack
    print('ground truth label is ', labels)
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    if opt.data == 'cifar100':
        shape = (3, 32, 32)
    elif opt.data == 'FashionMinist':
        shape = (1, 32, 32)

    if opt.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
    else:
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape) # specify label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm
    mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    print("after optimization, the true mse loss {}".format(mean_loss))

    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if opt.MoEx and opt.Mixup:
        method = 'MixupMoEx'+str(opt.alpha)
    elif opt.MoEx:
        method = 'MoEx'
    elif opt.Mixup:
        method = 'Mixup'+str(opt.alpha)
    else:
        method = ''
    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/{}_rec_{}_{}.jpg'.format(save_dir, idx,method,opt.aug_list))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/{}_ori.jpg'.format(save_dir, idx))


    test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    return {'test_mse': test_mse,
        'feat_mse': feat_mse,
        'test_psnr': test_psnr
    }




def create_checkpoint_dir():
    if opt.MoEx and opt.Mixup:
        return 'checkpoints/label/MixupMoex_alpha_{}_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.alpha,opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    elif opt.MoEx:
        return 'checkpoints/label/MoEx_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    elif opt.Mixup:
         return 'checkpoints/label/Mixup_alpha_{}_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.alpha,opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    else:
        return 'checkpoints/label/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)



def main():
    if opt.MoEx:
        print('MoEx mode')
    if opt.Mixup:
        print('Mixup mode')
    # if opt.add_defense:
    #     print(opt.defense)
    global trained_model
    print(opt)
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False
        
    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        # filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')
        filename = os.path.join(checkpoint_dir,f'{opt.arch}_{defs.epochs}.pth')
        print(filename)
        if not os.path.exists(filename):
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')

        print(filename)
        assert os.path.exists(filename)
        model.load_state_dict(torch.load(filename))

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()
    sample_list = [i for i in range(100)]
    metric_list = list()
    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        if idx < opt.resume:
            continue
        print('attach {}th in {}'.format(idx, opt.aug_list))
        metric = reconstruct(idx, model, loss_fn, trainloader, validloader)
        metric_list.append(metric)
    save_dir = create_save_dir()
    if opt.MoEx and opt.Mixup:
        method = 'MixupMoEx'+str(opt.alpha)
    elif opt.MoEx:
        method = 'MoEx_'+str(opt.alpha)
    elif opt.Mixup:
        method = 'Mixup'
    else:
        method = ''
    np.save('{}/metric_{}_{}.npy'.format(save_dir, method, opt.aug_list),metric_list)



if __name__ == '__main__':
    main()
