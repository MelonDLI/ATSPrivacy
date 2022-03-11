import preprocess
import inversefed
import network

import os


setup = inversefed.utils.system_startup()

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')

opt = parser.parse_args()

def create_checkpoint_dir():
    # return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    return 'checkpoints/ResNet20-4_50.pth'

def main():
    global trained_model
    print(opt)
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    # model = create_model(opt)
    model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, num_channels=num_channels, base_width=16 * 4)
    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False

    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

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
    np.save('{}/metric.npy'.format(save_dir), metric_list)



if __name__ == '__main__':
    main()
