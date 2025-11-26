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
python main.py --model qwen-7B --experiments all_rfh layerwise_rfh headwise_rfh topk_rfh
python main.py --model qwen-7B --experiments induction_heads
python main.py --model qwen-7B --experiments retrieval_heads
python main.py --model qwen-7B --experiments induction_and_retrieval_heads
python main.py --model qwen-7B --experiments induction_retrieval_and_rfh_heads
