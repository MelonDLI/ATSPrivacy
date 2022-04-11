import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def create_save_dir():
    return '/home/remote/u7076589/ATSPrivacy/MMD_result_epoch_psnr' #! change

def set_attack_model():
    # return '/home/remote/u7076589/ATSPrivacy/benchmark/images/data_cifar100_arch_ResNet20-4' #!change
    # return '/home/remote/u7076589/ATSPrivacy/benchmark/images/MMD_defense_prune_95_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
    # return '/home/remote/u7076589/ATSPrivacy/benchmark/images/MMD_defense_prune_99_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
    return '/home/remote/u7076589/ATSPrivacy/benchmark/images/MMD_defense_prune_95_first_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
    # return '/home/remote/u7076589/ATSPrivacy/benchmark/images/MMD_defense_prune_95_data_cifar100_arch_ResNet20-4_epoch_200_optim_inversed_mode_aug_auglist__rlabel_False/metric.npy'
def save_plot(stats,path):
    # psnr
    fig, ax = plt.subplots(figsize=[8,6])

    line1 = ax.plot(stats['psnr'], label='average psnr', color='red')
    line2 = ax.plot(stats['max_psnr'], label='max psnr', color='navy')  
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    ax.set_title('Attack epoch vs PSNR')

    lines = line1+line2
    labels = [_.get_label() for _ in lines]
    ax.legend(lines, labels, loc='center right')
    ax.set_xlim(xmin=1)
    fig.savefig('{}/epoch_psnr.png'.format(path))
    plt.close(fig)    # close the figure window

def main():
    stats = defaultdict(list)
    # save_dir = create_save_dir()
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #! MixupMoexDefense
    file = set_attack_model()

    result = np.load(file,allow_pickle=True)
    sum_ = 0
    max_psnr = 0
    for j in range(len(result)):
        sum_+=result[j]['test_psnr']
        if result[j]['test_psnr']>max_psnr:
            max_psnr = result[j]['test_psnr']
    print('----------------------------------')
    print('number of result:{}'.format(len(result)))
    print('ResNet:')
    print('average: {}'.format(sum_/len(result)))
    print('max:{}'.format(max_psnr))
    stats['psnr'].append(sum_/len(result))
    stats['max_psnr'].append(max_psnr)
    # np.save('{}/epoch_psnr.npy'.format(save_dir), stats)
    # save_plot(stats,save_dir)
    
if __name__ == '__main__':
    main()