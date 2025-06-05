#!/bin/sh
#SBATCH --job-name=perf
#SBATCH --output=./sout/perf/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-gpu=50G
#SBATCH --partition=gpu
#SBATCH --exclusive

srun python perf.py