#!/bin/bash
#SBATCH --job-name=Mixup             # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=2                              # Run on a single CPU
#SBATCH --mem=32gb                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=ATSprivacy_%j.log                 # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
echo "Mixup PSNR result"
# data=cifar100
# epoch=200
# arch='ResNet20-4'
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark/cifar100_train.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='' --mode=aug 
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark/cifar100_attack.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='' --mode=aug --optim='inversed'
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python -u searchalg/search_best.py --arch=ResNet20-4 --data=cifar100
# python /home/remote/u7076589/ATSPrivacy/result_metric_Mixup.py
echo "--- done ---"
date