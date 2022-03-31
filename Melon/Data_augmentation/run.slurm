#!/bin/bash
#SBATCH --job-name=ResNet_Data_augmentation              # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=2                              # Run on a single CPU
#SBATCH --mem=32gb                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=ResNet_without_DA_train%j.log                 # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
echo "ResNet20-4 without data augmentation training, save part path"
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif \
python ResNet_train_DA.py --aug_list=  --mode=aug --epochs=200
# echo "MoEx ResNet20-4 with data augmentation 43-18-18 training, save all path %10"
# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif \
# python MoEx_train_DA.py --aug_list=43-18-18 --mode=aug --epochs=200
echo "--- done ---"
date
