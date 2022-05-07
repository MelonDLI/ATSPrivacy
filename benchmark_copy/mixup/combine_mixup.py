import os
import cv2

if __name__ == '__main__':
    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/mixup_comb_100'
    # check the images
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup'
    image_list = [x for x in os.listdir(data_root)]
    print(f'images: {len(image_list)}')
    # read in images
    for i in range(100):
        ori=cv2.imread('{}/{}_ori.jpg'.format(data_root,i+1))

        mix_3 = cv2.imread('{}/{}_rec_Mixup_0.3_.jpg'.format(data_root,i+1))
        mix_4 = cv2.imread('{}/{}_rec_Mixup_0.4_.jpg'.format(data_root,i+1))
        mix_5 = cv2.imread('{}/{}_rec_Mixup_0.5_.jpg'.format(data_root,i+1))

        img_h=cv2.hconcat([ori,mix_3,mix_4,mix_5])
        
        cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i+1),img_h)

    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/mixup_comb_10'
    # check the images
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/mixup/mixup_comb_100'
    image_list = [x for x in os.listdir(data_root)]
    print(f'images: {len(image_list)}')
    # read in images
    i=2
    pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,1))
    while i<=100:
        ori=cv2.imread('{}/{}_concat.jpg'.format(data_root,i))
        
        pre=cv2.vconcat([pre,ori])

        if i%10==0:
            cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i),pre)
            i+=1
            pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,i))

        i+=1