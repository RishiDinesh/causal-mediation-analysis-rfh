#!/bin/bash
srun --partition gpunodes \
     --cpus-per-task=4 \
     --mem=30G \
     --gres=gpu:rtx_a2000:1 \
     patch_qwen_1q5B.sh

# srun --partition gpunodes \
#      --cpus-per-task=4 \
#      --mem=30G \
#      --gres=gpu:rtx_a4500:1 \
#      --pty bash --login
     # -t 60 \
     # --pty bash --login
# --pty interactive