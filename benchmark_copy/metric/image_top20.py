import cv2
if __name__ == '__main__':
    data_root = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/images_2'
    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_psnr_small_20'

    psnr_large_20=[86 ,51, 63, 89 ,59, 11, 65, 69 , 8 ,80 ,85, 64, 82, 77, 13, 34 , 4 ,54 ,43, 94]
    psnr_small_20=[84 ,32 ,14 ,30 ,22 ,16 ,29, 45, 76, 48 ,75 ,44 ,27, 55 ,24 ,47, 79 ,74 ,25 ,20]

    for i in psnr_small_20:
        ori=cv2.imread('{}/{}_ori.jpg'.format(data_root,i+1))
        res_DA=cv2.imread('{}/{}_rec__3-1-7+43-18-18.jpg'.format(data_root,i+1))
        mm_DA = cv2.imread('{}/{}_rec_MixupMoEx_3-1-7+43-18-18.jpg'.format(data_root,i+1))
        img_h=cv2.hconcat([ori,res_DA,mm_DA])
        cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i+1),img_h)

    save_dir = '/home/remote/u7076589/ATSPrivacy/benchmark_copy/comb_psnr_big_20'
    for i in psnr_large_20:
        ori=cv2.imread('{}/{}_ori.jpg'.format(data_root,i+1))
        res_DA=cv2.imread('{}/{}_rec__3-1-7+43-18-18.jpg'.format(data_root,i+1))
        mm_DA = cv2.imread('{}/{}_rec_MixupMoEx_3-1-7+43-18-18.jpg'.format(data_root,i+1))
        img_h=cv2.hconcat([ori,res_DA,mm_DA])
        cv2.imwrite('{}/{}_concat.jpg'.format(save_dir, i+1),img_h)