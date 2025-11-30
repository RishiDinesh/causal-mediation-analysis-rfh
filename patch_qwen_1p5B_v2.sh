#!/bin/bash
#SBATCH --job-name=patch-1p5B
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Running on: $HOSTNAME"
cd /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh
source ./.venv/bin/activate
python main.py --n 1 --model qwen-1p5B --experiments all_rfh layerwise_rfh headwise_rfh topk_rfh
# python main.py --model qwen-1p5B --experiments induction_heads
# python main.py --model qwen-1p5B --experiments retrieval_heads
# python main.py --model qwen-1p5B --experiments induction_and_retrieval_heads
# python main.py --model qwen-1p5B --experiments induction_retrieval_and_rfh_heads
