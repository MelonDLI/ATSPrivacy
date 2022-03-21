import numpy as np
file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/images_ResNet20_50/metric.npy'
# file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/images/metric.npy'
result = np.load(file,allow_pickle=True)
print('number of result:{}'.format(len(result)))
sum_ = 0
max_psnr = 0
for i in range(43):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print(sum_/43) #10.424027647972107
print(max_psnr)
