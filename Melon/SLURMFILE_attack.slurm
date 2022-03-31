#!/bin/bash
#SBATCH --job-name=ResNet_Data_augmentation              # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=4                              # Run on a single CPU
#SBATCH --mem=16G                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=attack_DA_ResNet_%j.log                 # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
# echo "ResNet20-4 with data augmentation 43-18-18 attack"
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif \
# python3 attack.py --model_path='/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/checkpoints/data_cifar100_arch_ResNet20-4_rlabel_False/ResNet20-4_200.pth' --save_dir='/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/benchmark/ResNet_DA'
echo "MoEx with data augmentation 43-18-18 attack"
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif \
python3 attack.py --model_path='/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/checkpoints/MoEx_bn_ResNet_20_lam_09_p_05_DA/ResNet20-4_200.pth' --save_dir='/home/remote/u7076589/ATSPrivacy/Melon/Data_augmentation/benchmark/MoEx_DA'
echo "--- done ---"
date
