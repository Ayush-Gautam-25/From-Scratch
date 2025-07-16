#!/bin/bash
#SBATCH --job-name=autoreg-train
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # HH:MM:SS
#SBATCH --partition=gpu             # Adjust based on your cluster
#SBATCH --account=pro00114885
# Load modules if needed
module purge
module load cuda/11.8
module load python/3.9

# Activate virtual environment
source ~/.bashrc
conda activate ecgfm

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Run your training script
python prepare_wikitext.py
