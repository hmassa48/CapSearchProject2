#!/usr/bin/env bash
#SBATCH --job-name=HyperChest
#SBATCH --partition=wacc
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --output=hyper_output-%j.txt
#SBATCH --gres=gpu:4


cd $SLURM_SUBMIT_DIR


module load anaconda/wml
bootstrap_conda

conda activate env697

python chestTune.py 

