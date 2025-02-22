#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l mem_free=32G
#$ -l h_rt=24:00:00
#$ -l gpu=2
#$ -pe smp 8
#$ -j y
#$ -o logs/train_$JOB_ID.log

# Load required modules
module load cuda/11.8
module load python/3.9

# Activate your conda environment
source activate falcon_env

# Run the training script
python train_falcon_wynton.py
