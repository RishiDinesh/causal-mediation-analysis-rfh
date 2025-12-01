#!/bin/bash
#SBATCH --job-name=patch-1p5B-test
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# bash -c "DEBUG=1 /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/run_patching.sh qwen-1p5B"
bash -c "/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/run_patching.sh qwen-1p5B"