#!/bin/bash
#SBATCH --job-name=SSIM             # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dan.li@anu.edu.au         # Where to send mail
#SBATCH --ntasks=2                              # Run on a single CPU
#SBATCH --mem=16G                               # Job memory request
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=teaching                        # Partition to schedule to
#SBATCH --gres=gpu:1     				        # Request a GPU allocation
#SBATCH --output=SSIM_%j.log                   # Standard output and error log
#SBATCH --qos=teachingproject

pwd; hostname; date
echo "--- starting ---"
# Job commands here
# echo "-- with search ---"
echo " SSIM ResNet original and reconstruction"

# singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/ssim_basis.py --data=cifar100 --arch='ResNet20-4' --epochs=200 --aug_list='3-1-7+43-18-18' --mode=aug --optim='inversed'  --MoEx=True --Mixup=True
singularity exec --nv /opt/apps/containers/pytorch_22.01-py3.sif python /home/remote/u7076589/ATSPrivacy/benchmark_copy/metric/result_metric.py
echo "--- done ---"
date
