#!/bin/bash
#SBATCH --job-name=MoEx_Data_augmentation              # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=2                              # Run on a single CPU
#SBATCH --mem=32gb                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=MoEx_DA_3-1-7_train_%j.log                 # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
echo "MoEx ResNet20-4 with data augmentation 3-1-7 training, save all path %10"
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif \
python ResNet_train_DA.py --aug_list=3-1-7 --mode=aug --epochs=200 --save_dir='checkpoints/MoEx_3-1-7' --MoEx=True
echo "--- done ---"
date