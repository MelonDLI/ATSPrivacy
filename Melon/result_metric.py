import numpy as np
file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/images/metric.npy'
# file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/images_ResNet20_200_2nd/metric.npy'
result = np.load(file,allow_pickle=True)
print('number of result:{}'.format(len(result)))
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

print('----------------------------------')
file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/images_ResNet20_200_1st/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('number of result:{}'.format(len(result)))
print('ResNet original:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))
