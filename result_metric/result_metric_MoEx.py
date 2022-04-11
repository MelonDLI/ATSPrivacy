import numpy as np

# #! MoEx 38-3-7+17-33-45
# file ='/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_38-3-7+17-33-45_rlabel_False/metric.npy'
# result = np.load(file,allow_pickle=True)
# sum_ = 0
# max_psnr = 0
# for i in range(len(result)):
#     sum_+=result[i]['test_psnr']
#     if result[i]['test_psnr']>max_psnr:
#         max_psnr = result[i]['test_psnr']
# print('----------------------------------')
# print('number of result:{}'.format(len(result)))
# print('ResNet+38-3-7+17-33-45:')
# print('average: {}'.format(sum_/len(result)))
# print('max:{}'.format(max_psnr))

# #! MoEx 17-33-45
# file ='/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_17-33-45_rlabel_False/metric.npy'
# result = np.load(file,allow_pickle=True)
# sum_ = 0
# max_psnr = 0
# for i in range(len(result)):
#     sum_+=result[i]['test_psnr']
#     if result[i]['test_psnr']>max_psnr:
#         max_psnr = result[i]['test_psnr']
# print('----------------------------------')
# print('number of result:{}'.format(len(result)))
# print('ResNet+17-33-45:')
# print('average: {}'.format(sum_/len(result)))
# print('max:{}'.format(max_psnr))

# #! MoEx 38-3-7
# file ='/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_38-3-7_rlabel_False/metric.npy'
# result = np.load(file,allow_pickle=True)
# sum_ = 0
# max_psnr = 0
# for i in range(len(result)):
#     sum_+=result[i]['test_psnr']
#     if result[i]['test_psnr']>max_psnr:
#         max_psnr = result[i]['test_psnr']
# print('----------------------------------')
# print('number of result:{}'.format(len(result)))
# print('ResNet+38-3-7:')
# print('average: {}'.format(sum_/len(result)))
# print('max:{}'.format(max_psnr))

#! MoEx 42-3-18+29-3-41
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_42-3-18+29-3-41_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx 42-3-18+29-3-41:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx 29-3-41
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_29-3-41_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx+29-3-41:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx 42-3-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_42-3-18_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx 42-3-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

# #! MoEx 43-18-18+3-1-7
file ='/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_43-18-18+3-1-7_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('MoEx+43-18-18+3-1-7:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx 3-1-7+43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_3-1-7+43-18-18_rlabel_False/metric.npy'
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

#! MoEx 43-18-18
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_43-18-18_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet+43-18-18:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx 3-1-7
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist_3-1-7_rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet+3-1-7:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))

#! MoEx
file = '/home/remote/u7076589/ATSPrivacy/benchmark/images/MoEx_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
result = np.load(file,allow_pickle=True)
sum_ = 0
max_psnr = 0
for i in range(len(result)):
    sum_+=result[i]['test_psnr']
    if result[i]['test_psnr']>max_psnr:
        max_psnr = result[i]['test_psnr']
print('----------------------------------')
print('number of result:{}'.format(len(result)))
print('ResNet:')
print('average: {}'.format(sum_/len(result)))
print('max:{}'.format(max_psnr))