#!/bin/bash
#SBATCH --job-name=transFURmer_300
#SBATCH --output=logs/transFURmer_%j.log
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# 1. Run the containerized training
timeout 715m apptainer exec --nv env.sif python src/experiments/$1.py