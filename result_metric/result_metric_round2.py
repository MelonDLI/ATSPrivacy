import numpy as np

#! Mixup+MoEx 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup+MoEx 3-1-7+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup+MoEx
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup+MoEx:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! Mixup 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_3-1-7+43-18-18.npy'
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


#! Mixup 
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('Mixup:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx+3-1-7+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MoEx_.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! ResNet
file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric__3-1-7+43-18-18.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet+3-1-7+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))