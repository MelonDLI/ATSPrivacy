import os
import cv2

if __name__ == '__main__':
    # save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_combination_images'
    # # check the images
    # data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/combination_images'
    # image_list = [x for x in os.listdir(data_root)]
    # print(f'images: {len(image_list)}')
    # # read in images
    # i=2
    # pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,1))
    # while i<=100:
    #     ori=cv2.imread('{}/{}_concat.jpg'.format(data_root,i))
        
    #     pre=cv2.vconcat([pre,ori])

    #     if i%10==0:
    #         cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i),pre)
    #         i+=1
    #         pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,i))

    #     i+=1
    data_root='/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_psnr_small_20'
    idx_0=84
    idx_list = [32 ,14 ,30 ,22 ,16 ,29, 45, 76, 48 ,75 ,44 ,27, 55 ,24 ,47, 79 ,74 ,25 ,20]
    pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,idx_0+1))
    # image_read=[]
    for idx in idx_list:
        # print(idx)
        ori=cv2.imread('{}/{}_concat.jpg'.format(data_root,idx+1))
        pre=cv2.vconcat([pre,ori])
    
    cv2.imwrite('{}/comb_psnr_small.jpg'.format(data_root),pre)

    data_root='/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_psnr_large_20'
    idx_0=86 
    pre = cv2.imread('{}/{}_concat.jpg'.format(data_root,idx_0+1))
    idx_list = [51, 63, 89 ,59, 11, 65, 69 , 8 ,80 ,85, 64, 82, 77, 13, 34 , 4 ,54 ,43, 94]
    for idx in idx_list:
        # print(idx)
        ori=cv2.imread('{}/{}_concat.jpg'.format(data_root,idx+1))
        pre=cv2.vconcat([pre,ori])
    
    cv2.imwrite('{}/comb_psnr_large.jpg'.format(data_root),pre)
