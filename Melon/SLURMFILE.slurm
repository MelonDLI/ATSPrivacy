#!/bin/bash
#SBATCH --job-name=epoch_200_cifar100_reconstr   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=teaching
#SBATCH --gres=gpu:1
#SBATCH --output=cifar100_recon_ResNet_epoch_200_%j.log   # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- training ---"
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python3 attack.py --model_path='/home/remote/u7076589/ATSPrivacy/Melon/checkpoints/data_cifar100_arch_ResNet20-4_rlabel_False/ResNet20-4_200.pth'
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python3 train.py --epochs=200
echo "--- done ---"
date
