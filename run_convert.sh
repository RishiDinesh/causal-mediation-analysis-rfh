#!/bin/bash
source /w/20252/wjcai/causal_inf/repo_switch.sh
cd /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh
source ./.venv/bin/activate

TRACE_DIR=/w/20252/wjcai/causal_inf/ACV/misc/R2A/output/reasoning_traces
OUTPUT_DIR=/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/data_v3

for SUBJECT in math physics; do
  python /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/process_traces_v2.py \
    --trace-dir "${TRACE_DIR}" \
    --subject "${SUBJECT}" \
    --output "${OUTPUT_DIR}/traces_${SUBJECT}.jsonl"
done
