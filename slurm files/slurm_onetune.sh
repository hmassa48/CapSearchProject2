#!/usr/bin/env bash
#SBATCH --job-name=HyperUNet
#SBATCH --partition=wacc
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=hyperLGG_output-%j.txt
#SBATCH --gres=gpu:4


cd $SLURM_SUBMIT_DIR

module load anaconda/wml
bootstrap_conda

conda activate env697

python one_tune.py
