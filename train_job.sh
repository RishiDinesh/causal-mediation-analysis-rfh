#!/bin/bash
#SBATCH --job-name=patch-7B
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=rishidinesh@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /w/nobackup/385/scratch-space/expires-2025-Nov-30/rishi/causal-mediation-analysis-rfh
source ./.venv/bin/activate
python run_optimized_patcher.py --model qwen-7B --experiments topk_rfh
