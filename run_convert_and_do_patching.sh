#!/bin/bash
source /w/20252/wjcai/causal_inf/repo_switch.sh
cd /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh
source ./.venv/bin/activate

python /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/notebooks/david/process_traces_v2.py \
  --trace-dir /w/20252/wjcai/causal_inf/ACV/misc/R2A/output/reasoning_traces \
  --output /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces.jsonl

python main.py \
  --n 1 \
  --model qwen-1p5B \
  --experiments all_rfh layerwise_rfh headwise_rfh topk_rfh \
  --traces_path /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces.jsonl \
  --output_suffix test_run
python main.py --model qwen-1p5B --experiments induction_heads
python main.py --model qwen-1p5B --experiments retrieval_heads
python main.py --model qwen-1p5B --experiments induction_and_retrieval_heads
python main.py --model qwen-1p5B --experiments induction_retrieval_and_rfh_heads