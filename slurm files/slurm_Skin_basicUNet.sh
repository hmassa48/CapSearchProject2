#!/usr/bin/env bash
#SBATCH --job-name=skinUNet
#SBATCH --partition=wacc
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --output=skinn_output-%j.txt
#SBATCH --gres=gpu:2


cd $SLURM_SUBMIT_DIR

module load anaconda/wml
bootstrap_conda

conda activate env697

python trainSkinUNet.py


