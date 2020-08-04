#!/usr/bin/env bash
#SBATCH --job-name=LGGUNet
#SBATCH --partition=wacc
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=LGG_output-%j.txt
#SBATCH --gres=gpu:2


cd $SLURM_SUBMIT_DIR

module load anaconda/wml
bootstrap_conda

conda activate env697


python trainBasicUNet.py


