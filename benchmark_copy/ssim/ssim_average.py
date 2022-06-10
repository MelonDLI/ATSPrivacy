import numpy as np

import numpy as np

#! ResNet
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec__.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! ResNet 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec__3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet  3-1-7+43-18-18:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! MoEx
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_MoEx_.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! MoEx 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_MoEx_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx  3-1-7+43-18-18:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! Mixup
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_Mixup_.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! Mixup 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_Mixup_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup  3-1-7+43-18-18:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))


#! MoEx+Mixup
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_MixupMoEx_.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MixupMoEx:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))

#! MixupMoEx 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/ssim/metric_ssim_whirl_ori_rec_MixupMoEx_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ssim = 0
max_ssim = 0
for i in range(len(result)):
    sum_ssim+=result[i][1]
    if result[i][1]>max_ssim:
        max_ssim = result[i][1]
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MixupMoEx  3-1-7+43-18-18:')
print('average: {}'.format(sum_ssim/len(result)))
print('max:{}'.format(max_ssim))