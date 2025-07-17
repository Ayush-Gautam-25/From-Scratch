#!/bin/bash
#SBATCH --job-name=autoreg-exp
#SBATCH --output=logs4.out
#SBATCH --error=logs4.err
#SBATCH --nodes=1               # or more if you want multi-node
#SBATCH --ntasks-per-node=2     # 2 tasks per node to match 2 GPUs
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # HH:MM:SS
#SBATCH --partition=gpu             # Adjust based on your cluster
#SBATCH --account=pro00114885
# Load modules if needed
module load cuda12.0


# Activate virtual environment
source ~/.bashrc
conda activate ecgfm

# Set CUDA paths
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
echo "$MASTER_ADDR"
export MASTER_PORT=12355  # Make sure it's open or unused
export WORLD_SIZE=$SLURM_NTASKS
export NODE_RANK=$SLURM_NODEID

# Launch distributed training using torchrun or mpirun
torchrun \
  --nproc_per_node=2 \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
    ddp_train_model.py
