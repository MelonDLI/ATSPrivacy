import numpy as np

#! Mixup 0.3
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/metric_Mixup_0.3_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mix up 0.3:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup 0.4
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/metric_Mixup_0.4_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mix up 0.4:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup 0.5
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/metric_Mixup_0.5_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mix up 0.5:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))