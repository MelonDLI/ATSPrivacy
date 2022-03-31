import numpy as np


file = '/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/benchmark/MoEx_DA/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx+DA:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

# file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/MoEx_bn_update_images/metric.npy'  # lambda=0.5
file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/MoEx_bn_lambda_09/metric.npy'
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


print('----------------------------------')
file = '/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/benchmark/ResNet_DA/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('number of result:{}'.format(len(result)))
print('ResNet+DA:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))


# print('----------------------------------')
# file = '/home/remote/u7076589/ATSPrivacy/Melon/benchmark/MoEx_bn_images/metric.npy'
# result = np.load(file,allow_pickle=True)
# sum_ = 0
# max_psnr = 0
# for i in range(len(result)):
#     sum_+=result[i]['test_psnr']
#     if result[i]['test_psnr']>max_psnr:
#         max_psnr = result[i]['test_psnr']
# print('number of result:{}'.format(len(result)))
# print('MoEx bn:')
# print('average: {}'.format(sum_/len(result)))
# print('max:{}'.format(max_psnr))

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
