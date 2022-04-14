import os
import cv2

if __name__ == '__main__':
    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/combination_images'
    # check the images
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2'
    image_list = [x for x in os.listdir(data_root)]
    print(f'images: {len(image_list)}')
    # read in images
    for i in range(100):
        ori=cv2.imread('{}/{}_ori.jpg'.format(data_root,i+1))

        res = cv2.imread('{}/{}_rec__3-1-7+43-18-18.jpg'.format(data_root,i+1))

        mixup=cv2.imread('{}/{}_rec_Mixup_.jpg'.format(data_root,i+1))
        mixup_DA=cv2.imread('{}/{}_rec_Mixup_3-1-7+43-18-18.jpg'.format(data_root,i+1))

        moex=cv2.imread('{}/{}_rec_MoEx_.jpg'.format(data_root,i+1))
        moex_DA=cv2.imread('{}/{}_rec_MoEx_3-1-7+43-18-18.jpg'.format(data_root,i+1))

        mm=cv2.imread('{}/{}_rec_MixupMoEx_.jpg'.format(data_root,i+1))
        mm_DA=cv2.imread('{}/{}_rec_MixupMoEx_3-1-7+43-18-18.jpg'.format(data_root,i+1))

        img_h=cv2.hconcat([ori,res,mixup,mixup_DA,moex,moex_DA,mm,mm_DA])
        
        cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i+1),img_h)