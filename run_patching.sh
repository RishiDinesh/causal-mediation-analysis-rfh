#!/bin/bash
source /w/20252/wjcai/causal_inf/repo_switch.sh
cd /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh
source ./.venv/bin/activate

run_main() {
  # echo "\n"
  # echo "[run_main] python main.py $*"
  # echo "\n"
  if [[ "${DEBUG:-}" == "1" || "${DEBUG:-}" == "true" ]]; then
    python main.py "$@" --n 1
  else
    python main.py "$@"
  fi
}

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("qwen-7B" "qwen-1p5B")
fi

declare -A SUBJECT_SETTINGS=(
  ["math"]="/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces_math.jsonl college_math"
  ["physics"]="/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3/traces_physics.jsonl college_physics"
)

for model in "${MODELS[@]}"; do
  for subject in "${!SUBJECT_SETTINGS[@]}"; do
    read -r traces_path output_suffix <<<"${SUBJECT_SETTINGS[$subject]}"
    run_main \
      --model "${model}" \
      --experiments all_rfh layerwise_rfh headwise_rfh topk_rfh \
      --traces_path "${traces_path}" \
      --output_suffix "${output_suffix}"
    
    run_main --model "${model}" --experiments induction_heads --traces_path "${traces_path}" --output_suffix "${output_suffix}"
    run_main --model "${model}" --experiments retrieval_heads --traces_path "${traces_path}" --output_suffix "${output_suffix}"
    run_main --model "${model}" --experiments induction_and_retrieval_heads --traces_path "${traces_path}" --output_suffix "${output_suffix}"
    run_main --model "${model}" --experiments induction_retrieval_and_rfh_heads --traces_path "${traces_path}" --output_suffix "${output_suffix}"
  done
done
