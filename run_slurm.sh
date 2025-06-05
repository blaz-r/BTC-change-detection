#!/bin/sh
#SBATCH --job-name=btc
#SBATCH --output=./sout/sb/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --partition=gpu

echo "$@"

srun --output="./sout/%j.out" python train.py --config configs/exp/BTC-B.yaml "$@" --devices -1