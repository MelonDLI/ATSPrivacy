#!/bin/bash
#SBATCH --job-name=Data_augmentation              # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=2                              # Run on a single CPU
#SBATCH --mem=32gb                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=ATSprivacy_ResNet_3-1-7+43-18-18_%j.log                 # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
echo "3-1-7+43-18-18 privacy-preserving evaluation"
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark/cifar100_train.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='3-1-7+43-18-18' --mode=aug 
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark/cifar100_attack.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='3-1-7+43-18-18' --mode=aug --optim='inversed'
echo "--- done ---"
date