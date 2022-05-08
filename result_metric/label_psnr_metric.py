import numpy as np

#! label corrupted
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/corrupt_label_image/metric__.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('label corrupted:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))