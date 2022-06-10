import numpy as np

#! ResNet 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_3-1-7+43-18-18_rlabel_False/metric.npy'
result_DA = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_.npy'
result_mix = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_3-1-7+43-18-18.npy'
result_mix_da = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_.npy'
result_moex = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_3-1-7+43-18-18.npy'
result_moex_da = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_.npy'
result_mm = np.load(file,allow_pickle=True)
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
result_mm_da = np.load(file,allow_pickle=True)

for i in range(100):
    a=result[i]['test_psnr']
    b=result_DA[i]['test_psnr']
    c=result_mix[i]['test_psnr']
    d=result_mix_da[i]['test_psnr']
    e=result_moex[i]['test_psnr']
    f=result_moex_da[i]['test_psnr'] 
    g=result_mm[i]['test_psnr']
    h=result_mm_da[i]['test_psnr']
    
    # print('{} {} {} {} {} {} {} {}'.format(a,b,c,d,e,f,g,h))
    print(g)


