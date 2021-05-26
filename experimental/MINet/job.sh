#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:2
#SBATCH --time=48:00:00

export PATH=/home/jc3/miniconda2/bin/:$PATH
source activate pytorch-1.5.0


python train.py