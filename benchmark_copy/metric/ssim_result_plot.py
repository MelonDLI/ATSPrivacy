import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# import argparse
# parser = argparse.ArgumentParser(description='load metric.')
# parser.add_argument('--ResNet', default=False, type=bool, help='ResNet or not.')
# parser.add_argument('--MM', default=False, type=bool, help='Mixup + MoEx or not.')
# parser.add_argument('--Mixup', default=False, type=bool, help='Mixup or not.')
# parser.add_argument('--MoEx', default=False, type=bool, help='MoEx or not.')

# opt = parser.parse_args()

def title_assign(type):
    if type=='ResNet':
        return 'ResNet+DA'
    if type=='MM':
        return 'Mixup+MoEx+DA'
    if type=='Mixup':
        return 'Mixup+DA'
    if type=='MoEx':
        return 'MoEx+DA'
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_3-1-7+43-18-18.npy'
    # title = 'Mixup+DA'
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_3-1-7+43-18-18.npy'
    # title = 'MoEx+DA'
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
    # title = 'Mixup+MoEx+DA'

def psnr_file_path(type):
    if type=='ResNet':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric__3-1-7+43-18-18.npy'
    if type=='MM':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
    if type=='Mixup':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_3-1-7+43-18-18.npy'
    if type=='MoEx':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_3-1-7+43-18-18.npy'

def ssim_file_path(type):
    if type=='ResNet':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_blank__3-1-7+43-18-18.npy'
    if type=='MM':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_blank_MixupMoEx_3-1-7+43-18-18.npy'
    if type=='Mixup':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_blank_Mixup_3-1-7+43-18-18.npy'
    if type=='MoEx':
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_blank_MoEx_3-1-7+43-18-18.npy'

def ssim_recon_path(type):
    if type == 'ResNet':
        # return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_ResNet_ori_rec__3-1-7+43-18-18.npy'
        # return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_chess_ori_rec__3-1-7+43-18-18.npy'
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_white_ori_rec__3-1-7+43-18-18.npy'
    if type == 'MM':
        # return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_MM_ori_rec_MixupMoEx_3-1-7+43-18-18.npy'
        # return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_chess_ori_rec_MixupMoEx_3-1-7+43-18-18.npy'
        return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup/metric_ssim_white_ori_rec_MixupMoEx_3-1-7+43-18-18.npy'
def file_load(filename):
    # ResNet
    file = filename
    return np.load(file,allow_pickle=True)

def create_save_dir():
    return '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/mixup'
# TODO: feature
    # feature
    # feature_path = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/basis_metric__.npy'
    # feature = np.load(feature_path)
    # print(feature)

if __name__ == '__main__':
    save_dir = create_save_dir()
    type='ResNet'
    title=title_assign(type)
    # psnr = file_load(psnr_file_path(type))
    # ssim = file_load(ssim_file_path(type))
    ssim_recon = file_load(ssim_recon_path(type))

    type='MM'
    title_MM=title_assign(type)
    # psnr_MM = file_load(psnr_file_path(type))
    # ssim_MM = file_load(ssim_file_path(type))
    ssim_recon_MM = file_load(ssim_recon_path(type))

    plot = {'Res_ori':[],'Res_recon':[],'MM_ori':[],'MM_recon':[]}

    for i in range(100):
        plot['Res_ori'].append(ssim_recon[i][0])
        plot['Res_recon'].append(ssim_recon[i][1])
        plot['MM_ori'].append(ssim_recon_MM[i][0])
        plot['MM_recon'].append(ssim_recon_MM[i][1])
    
    plt.scatter(plot['Res_ori'],plot['Res_recon'],label='ResNet+DA')
    plt.scatter(plot['MM_ori'],plot['MM_recon'],label='MM+DA')

    plt.title('SSIM vs SSIM with white as ref')
    plt.ylabel('SSIM: ori vs rec')
    # plt.ylim(ymin=5,ymax=20)
    plt.xlabel('SSIM')
    # plt.xlim(xmin=0,xmax=0.05)
    plt.legend(loc="upper right")
    plt.savefig('{}/ssim_ResNet_MM_white.jpg'.format(save_dir))
    plt.close()
