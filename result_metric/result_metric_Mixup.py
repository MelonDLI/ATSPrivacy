import numpy as np

#! Mixup_lambda=0.1
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2/metric_Mixup_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup_lambda=0.1:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup_alpha0.1-beta10 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2/metric_Mixup_alpha0.1_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup_alpha+3-1-7+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/Mixup_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_3-1-7+43-18-18_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup+3-1-7+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup_ 43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/Mixup_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_43-18-18_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup_+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup_ 3-1-7
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/Mixup_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_3-1-7_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup_+3-1-7:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup_
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/Mixup_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup_:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))