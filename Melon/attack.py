from preprocess import preprocess,create_config
import utils
import Models
from optimization_strategy import training_strategy
from reconstruction_algorithm import GradientReconstructor
import metrics
import torchvision
import torch
import os
import numpy as np

num_images = 1
setup = utils.system_startup()
# TODO opt
# config = create_config(opt)
config = create_config() 
# cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
# cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]

def create_save_dir():
    # return 'benchmark/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
    #     opt.aug_list, opt.rlabel)
    return 'benchmark/images/'

def reconstruct(idx, model, loss_fn, trainloader, validloader):
    # if opt.data=='cifar100':
    dm = torch.as_tensor(cifar100_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar100_std, **setup)[:, None, None]

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
    rec_machine = GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    # if opt.data == 'cifar100':
    shape = (3, 32, 32)
    # elif opt.data == 'FashionMinist':
        # shape = (1, 32, 32)
    # if opt.rlabel:
    output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm
    mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    print("after optimization, the true mse loss {}".format(mean_loss))

    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/rec_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/ori_{}.jpg'.format(save_dir, idx))


    test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
    test_psnr = metrics.psnr(output_denormalized, input_denormalized)
    print('test_mse: {}'.format(test_mse))
    print('feat_mse: {}'.format(feat_mse))
    print('test_psnr: {}'.format(test_psnr))
    return {'test_mse': test_mse,
        'feat_mse': feat_mse,
        'test_psnr': test_psnr
    }
def main():
    # global trained_model=True
    defs = training_strategy('conservative'); defs.epochs = 100
    loss_fn, trainloader, validloader = preprocess(defs=defs)

    model = Models.ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=100, num_channels=3, base_width=16 * 4)
    model.to(**setup)
    # if trained_model:
    
    # reload trained model
    filename = 'checkpoints/ResNet20-4_50.pth'
    assert os.path.exists(filename)
    model.load_state_dict(torch.load(filename))

    model.eval()
    
    sample_list = [i for i in range(100)]
    metric_list = list()
    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        metric = reconstruct(idx, model, loss_fn, trainloader, validloader)
        metric_list.append(metric)
    save_dir = create_save_dir()
    np.save('{}/metric.npy'.format(save_dir), metric_list)

if __name__ == '__main__':
    main()