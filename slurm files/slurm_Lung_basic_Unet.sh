#!/usr/bin/env bash
#SBATCH --job-name=Lung_UNet
#SBATCH --partition=wacc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2-00:00:00
#SBATCH --output=Lung_output-%j.txt
#SBATCH --gres=gpu:2


cd $SLURM_SUBMIT_DIR

module load anaconda/wml
bootstrap_conda

conda activate ece6972


python trainLungUNet.py

