#!/bin/bash
 #SBATCH --job-name=reconstruction    # Job name
 #SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
 #SBATCH --mail-user=dan.li@anu.edu.au     # Where to send mail
 #SBATCH --ntasks=1                    # Run on a single CPU
 #SBATCH --mem=32gb                     # Job memory request
 #SBATCH --time=02:00:00               # Time limit hrs:min:sec
 #SBATCH --partition=teaching
 #SBATCH --gres=gpu:1
 #SBATCH --output=reconstruction_%j.log   # Standard output and error log
 #SBATCH --qos=teachingproject

 pwd; hostname; date
 echo "--- training ---"
 singularity exec --nv
/opt/apps/containers/tensorflow_21.10-tf2-py3.sif python3 test.py
 echo "--- done ---"
 date
