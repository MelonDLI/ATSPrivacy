import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric__3-1-7+43-18-18.npy'
    # title = 'ResNet+DA'
    file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_Mixup_3-1-7+43-18-18.npy'
    title = 'Mixup+DA'
    # file = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/metric_MixupMoEx_3-1-7+43-18-18.npy'
    # title = 'Mixup+MoEx+DA'
    result = np.load(file,allow_pickle=True)
    # sum_ = 0
    # max_psnr = 0
    # for i in range(len(result)):
    #     sum_+=result[i]['test_psnr']
    #     if result[i]['test_psnr']>max_psnr:
    #         max_psnr = result[i]['test_psnr']
    # print('----------------------------------')
    # print('number of result:{}'.format(len(result)))
    # print('ResNet+3-1-7+43-18-18:')
    # print('average: {}'.format(sum_/len(result)))
    # print('max:{}'.format(max_psnr))
    # print(result)

    plot = {'test_mse':[],'feat_mse':[],'test_psnr':[]}
    for i in range(len(result)):
        # if result[i]['feat_mse'].cpu().detach().numpy()<7.5: # MixupMoEx
        #     plot['test_mse'].append(result[i]['test_mse'])
        #     plot['feat_mse'].append(result[i]['feat_mse'].cpu().detach().numpy())
        #     plot['test_psnr'].append(result[i]['test_psnr'])
        plot['test_mse'].append(result[i]['test_mse'])
        plot['feat_mse'].append(result[i]['feat_mse'].cpu().detach().numpy())
        plot['test_psnr'].append(result[i]['test_psnr'])

    # # colors = np.random.rand(len(result))

    plt.scatter(plot['test_mse'], plot['feat_mse'])
    plt.xlabel('test mse')
    plt.ylabel('feat mse')
    plt.title(title)
    plt.savefig('mse_{}.jpg'.format(title),format='jpg')
    plt.close()

    plt.scatter(plot['feat_mse'],plot['test_psnr'])
    plt.ylabel('test psnr')
    plt.xlabel('feat mse')
    plt.title(title)
    plt.savefig('psnr_feat_mse_{}.jpg'.format(title),format='jpg')
    plt.close()
    
    plt.scatter(plot['test_mse'],plot['test_psnr'])
    plt.ylabel('test psnr')
    plt.xlabel('test_mse')
    plt.title(title)
    plt.savefig('psnr_test_mse_{}.jpg'.format(title),format='jpg')
    plt.close()