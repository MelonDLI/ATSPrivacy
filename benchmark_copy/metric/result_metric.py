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
    psnr = file_load(psnr_file_path(type))
    ssim = file_load(ssim_file_path(type))

    type='MM'
    title_MM=title_assign(type)
    psnr_MM = file_load(psnr_file_path(type))
    ssim_MM = file_load(ssim_file_path(type))

    plot = {'test_mse':[],'feat_mse':[],'test_psnr':[],'ssim':[],
            'test_mse_MM':[],'feat_mse_MM':[],'test_psnr_MM':[],'ssim_MM':[]}

    for i in range(100):
        # if result[i]['feat_mse'].cpu().detach().numpy()<7.5: # MixupMoEx
        # if result[i]['feat_mse'].cpu().detach().numpy()<3: #Mixup
        # if result[i]['feat_mse'].cpu().detach().numpy()<2: #MoEx
        # if result[i]['feat_mse'].cpu().detach().numpy()<20: #ResNet
        #     plot['test_mse'].append(result[i]['test_mse'])
        #     plot['feat_mse'].append(result[i]['feat_mse'].cpu().detach().numpy())
        #     plot['test_psnr'].append(result[i]['test_psnr'])
        plot['test_mse'].append(psnr[i]['test_mse'])
        plot['feat_mse'].append(psnr[i]['feat_mse'].cpu().detach().numpy())
        plot['test_psnr'].append(psnr[i]['test_psnr'])
        # plot['feat_var'].append(feature[i])
        # plot['mmd_psnr'].append(MMD[i]['test_psnr'])
        plot['ssim'].append(ssim[i])
        #MM
        plot['test_mse_MM'].append(psnr_MM[i]['test_mse'])
        plot['feat_mse_MM'].append(psnr_MM[i]['feat_mse'].cpu().detach().numpy())
        plot['test_psnr_MM'].append(psnr_MM[i]['test_psnr'])
        plot['ssim_MM'].append(ssim_MM[i])
    
    ################################################################
    # PSNR vs SSIM
    ################################################################
    # plt.scatter(plot['test_psnr'], plot['ssim'])
    # plt.xlabel('test psnr')
    # plt.ylabel('SSIM')
    # plt.xlim(xmin=0,xmax=20)
    # plt.title(title)
    # plt.savefig('psnr_ssim_{}.jpg'.format(title),format='jpg')
    # plt.close()

    plt.scatter(plot['ssim'],plot['test_psnr'],label='ResNet+DA')
    # plt.plot(plot['test_psnr'],plot['ssim'])
    plt.scatter(plot['ssim_MM'],plot['test_psnr_MM'],label='MM+DA')
    # plt.plot(plot['test_psnr_MM'],plot['ssim_MM'])
    plt.title('PSNR vs SSIM')
    plt.ylabel('PSNR')
    plt.ylim(ymin=5,ymax=20)
    plt.xlabel('SSIM')
    plt.xlim(xmin=0)
    plt.legend(loc="upper right")
    plt.savefig('{}/psnr_ssim_blank.jpg'.format(save_dir))
    plt.close()


    # # colors = np.random.rand(len(result))

    # plt.scatter(plot['test_mse'], plot['feat_mse'])
    # plt.xlabel('test mse')
    # plt.ylabel('feat mse')
    # plt.title(title)
    # plt.savefig('mse_{}.jpg'.format(title),format='jpg')
    # plt.close()

    # plt.scatter(plot['feat_mse'],plot['test_psnr'])
    # plt.ylabel('test psnr')
    # plt.xlabel('feat mse')
    # plt.title(title)
    # plt.savefig('psnr_feat_mse_{}.jpg'.format(title),format='jpg')
    # plt.close()
    # print(stats.pearsonr(plot['feat_mse'],plot['test_psnr']))
    
    # plt.scatter(plot['test_mse'],plot['test_psnr'])
    # plt.ylabel('test psnr')
    # plt.xlabel('test_mse')
    # plt.title(title)
    # plt.savefig('psnr_test_mse_{}.jpg'.format(title),format='jpg')
    # plt.close()

    #################################################################
    # feature variance for ResNet20 
    #! need unnormalized feature
    #################################################################
    # plt.scatter(plot['feat_var'],plot['test_mse'])
    # plt.xlabel('feature variance')
    # plt.ylabel('test mse')
    # plt.title(title)
    # plt.savefig('mse_feat_var_{}.jpg'.format(title),format='jpg')
    # plt.close()
    # print(stats.pearsonr(plot['feat_var'],plot['test_mse']))

    ################################################################
    # sort PSNR
    ################################################################
    # top_2_idx = np.argsort(plot['test_psnr']) # Resnet
    # print(top_2_idx)

    # print("ResNet+DA, MMD+DA")
    # for i in top_2_idx:
    #     print(' {}, {}'.format(plot['test_psnr'][i],plot['mmd_psnr'][i]))

