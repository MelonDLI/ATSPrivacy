import os
import cv2

if __name__ == '__main__':
    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_combination_images'
    # check the images
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/combination_images'
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