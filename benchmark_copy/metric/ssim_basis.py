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
import torch.utils.data as data
from torch.utils.data import DataLoader
from os.path import join
import numpy as np
from torch.autograd import Variable

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

from ssim import *
# from pytorch_ssim import *
# python /home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/basis.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='3-1-7+43-18-18' --mode=aug --optim='inversed'

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

def data_root():
    return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2'

def is_ori_image(filename):
        return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"]) and ('ori' in filename)

class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, image_list_rec, root_path):
        super(ImageDatasetFromFile, self).__init__()
        self.root_path = root_path
        self.image_filenames = image_list
        self.image_list_rec = image_list_rec
        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(join(self.root_path, self.image_filenames[index]))
        img = self.input_transform(img)
        if self.image_list_rec is not None:
            img_rec = Image.open(join(self.root_path,self.image_list_rec[index]))
            img_rec = self.input_transform(img_rec)
            return img, img_rec
        else:
            return img

    def __len__(self):
        return len(self.image_filenames)

def create_checkpoint_dir():
    if opt.MoEx and opt.Mixup:
        return 'checkpoints/MixupMoex_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    elif opt.MoEx:
        return 'checkpoints/MoEx_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    elif opt.Mixup:
         return 'checkpoints/Mixup_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    else:
        return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)

def create_save_dir():
    return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/'

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
    
    model.eval()
    sample_list = [i for i in range(100)]
    metric_list = list()
    
    # prepare data
    data_path = data_root()

    # image_list = [x for x in os.listdir(data_path) if is_ori_image(x)]
    # original images:
    image_list = ['{}_ori.jpg'.format(i+1) for i in range(100) ]
    # image_list = ['test_dog.jpg']
    print(image_list)
    print(f'images: {len(image_list)}')
    # ResNet Reconstruction images
    image_list_rec = ['{}_rec__.jpg'.format(i+1) for i in range(100) ]
    
    image_list_0= ['blank.jpg']
    data_set_0 = ImageDatasetFromFile(image_list_0, None, '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim')
    val_data_0 = DataLoader(data_set_0,batch_size=1,shuffle=False)
    img0 = next(iter(val_data_0))
    
    data_set = ImageDatasetFromFile(image_list, image_list_rec, data_path)
    val_data = DataLoader(data_set,batch_size=10,shuffle=False)
    i=1
    ssim_metric_list = list()

    for img,img_rec in val_data:
        ###################
        #mixup
        ##################
        rand_index = torch.randperm(inputs.size()[0]).cuda()
        target_a = targets
        target_b = targets[rand_index]
        input_a_var = torch.autograd.Variable(inputs, requires_grad=True)
        input_b_var = torch.autograd.Variable(inputs[rand_index], requires_grad=True)
        target_a_var = torch.autograd.Variable(target_a)
        target_b_var = torch.autograd.Variable(target_b)

        print(i)
        i+=1
        ##################
        # SSIM
        ##################
        # print(img.shape)
        ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
        # ssim_loss = 1 - ssim_module(img, img_rec)
        ssim_loss = 1 - ssim_module(img, img0)
        print(ssim_loss)
        # # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        # ssim_val = ssim(img, img_rec, data_range=255, size_average=False)
        # print(ssim_val)
        # ssim_loss = 1 - ssim(img, img_rec, data_range=255, size_average=True) # return a scalar
        # # ms_ssim_loss = 1 - ms_ssim( img, img, data_range=255, size_average=True )
        # print(ssim_loss)
        # print(ms_ssim_loss)
        ssim_metric_list.append(ssim_loss)
        
    save_dir=create_save_dir()
    if opt.MoEx and opt.Mixup:
        method = 'MixupMoEx'
    elif opt.MoEx:
        method = 'MoEx'
    elif opt.Mixup:
        method = 'Mixup'
    else:
        method = ''
    np.save('{}/metric_ssim_blank_{}_{}.npy'.format(save_dir,method,opt.aug_list),ssim_metric_list)
    
    # reuse the gaussian kernel with SSIM & MS_SSIM. 
# ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

# ssim_loss = 1 - ssim_module(X, Y)
# ms_ssim_loss = 1 - ms_ssim_module(X, Y)

    # # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # # Y: (N,3,H,W)  

    # # calculate ssim & ms-ssim for each image
    # ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
    # ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)
    
    
    # import statistics

    # for img in val_data:
    #     feature = model(img.to(**setup))
    #     # print(feature.detach().cpu().numpy())
    #     temp = feature.detach().cpu().numpy()
    #     # print(temp.reshape(-1))
    #     output = statistics.variance(temp.reshape(-1).tolist()) 

    #     # metric_list.append(feature.detach().cpu().mean())
    #     metric_list.append(output)
    #     # print(output)


    # # save data
    # if opt.MoEx and opt.Mixup:
    #     method = 'MixupMoEx'
    # elif opt.MoEx:
    #     method = 'MoEx'
    # elif opt.Mixup:
    #     method = 'Mixup'
    # else:
    #     method = ''
    # save_dir = create_save_dir()
    # np.save('{}/basis_metric_{}_{}.npy'.format(save_dir, method, opt.aug_list),metric_list)

def mixup_data(x, i, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    print(lam)
    batch_size = x.size()[0]

    index = torch.randperm(batch_size)
    index = index.long()

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # save mixup image
    save_dir = create_save_dir()
    # inputs = map(Variable, (mixed_x))
    torchvision.utils.save_image(mixed_x.cpu().clone(), '{}/mixup_{}.jpg'.format(save_dir,i))


if __name__ == '__main__':
    main()