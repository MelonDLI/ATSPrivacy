import torch
import os
import cv2
import torchvision
import numpy as np

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        # if mse > 0 and torch.isfinite(mse):
        #     return (10 * torch.log10(factor**2 / mse))
        # elif not torch.isfinite(mse):
        #     return img_batch.new_tensor(float('nan'))
        # else:
        #     return img_batch.new_tensor(float('inf'))
        if mse > 0:
            return (10 * np.log10(factor**2 / mse))
        elif mse == 0:
            print(0)
    if batched:
        psnr = get_psnr(img_batch, ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


if __name__ == '__main__':
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2'
    ori=cv2.imread('{}/{}_ori.jpg'.format(data_root,100))
    ref=cv2.imread('{}/{}_rec_Mixup_.jpg'.format(data_root,100))
    psnr_ori = psnr(ori,ori,batched=True, factor=255)
    print(psnr_ori)
    

    # #! Mixup+MoEx 3-1-7+43-18-18
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
    # result = np.load(file,allow_pickle=True)
    # sum_ = 0
    # max_psnr = 0
    # for i in range(len(result)):
    #     sum_+=result[i]['test_psnr']
    #     print(result[i]['test_psnr'])
    #     if result[i]['test_psnr']>max_psnr:
    #         max_psnr = result[i]['test_psnr']
    # print('----------------------------------')
    # print('number of result:{}'.format(len(result)))
    # print('Mixup+MoEx 3-1-7+43-18-18:')
    # print('average: {}'.format(sum_/len(result)))
    # print('max:{}'.format(max_psnr))