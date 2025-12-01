#!/bin/bash
source /w/20252/wjcai/causal_inf/repo_switch.sh
cd /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh
source ./.venv/bin/activate

run_main() {
  if [[ "${DEBUG:-}" == "1" || "${DEBUG:-}" == "true" ]]; then
    python main.py "$@" --n 1
  else
    python main.py "$@"
  fi
}

python /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/notebooks/david/process_traces_v2.py \
  --trace-dir /w/20252/wjcai/causal_inf/ACV/misc/R2A/output/reasoning_traces \
  --output /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces.jsonl

MODELS=("qwen-7B" "qwen-1p5B")
OUTPUT_SUFFIXES=("college_math" "college_physics")

for model in "${MODELS[@]}"; do
  for suffix in "${OUTPUT_SUFFIXES[@]}"; do
    run_main \
      --model "${model}" \
      --experiments all_rfh layerwise_rfh headwise_rfh topk_rfh \
      --traces_path /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces.jsonl \
      --output_suffix "${suffix}"
  done

  run_main --model "${model}" --experiments induction_heads
  run_main --model "${model}" --experiments retrieval_heads
  run_main --model "${model}" --experiments induction_and_retrieval_heads
  run_main --model "${model}" --experiments induction_retrieval_and_rfh_heads
done
