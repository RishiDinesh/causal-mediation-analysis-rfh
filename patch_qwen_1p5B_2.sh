#!/bin/bash
#SBATCH --job-name=patch-IRR-1p5B
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=rishidinesh@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /w/nobackup/385/scratch-space/expires-2025-Dec-09/rishi/causal-mediation-analysis-rfh
source ./.venv/bin/activate
python main.py --model qwen-1p5B --experiments induction_heads
python main.py --model qwen-1p5B --experiments retrieval_heads
python main.py --model qwen-1p5B --experiments induction_and_retrieval_heads
python main.py --model qwen-1p5B --experiments induction_retrieval_and_rfh_heads