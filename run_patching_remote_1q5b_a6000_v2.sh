#!/bin/bash
#SBATCH --job-name=patch-7B
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL
bash -c "/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/run_patching.sh qwen-1p5B"