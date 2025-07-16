#!/bin/bash
#SBATCH --job-name=autoreg-exp
#SBATCH --output=autoreg_tiny_result.out
#SBATCH --error=error.err
#SBATCH --ntasks=5
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
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Run your training script
python autoreg_tiny_generate.py
