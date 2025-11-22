import os
import json
import torch
import pandas as pd
from pathlib import Path
from src._types import PatchConfig
from src.constants import TOP_RFHS_BY_LAYER_HEAD, MODELS_LITERAL
from src.optimized_patcher import OptimizedActivationPatcher, clear_memory

MODEL_ALIAS: MODELS_LITERAL = "qwen-1p5B"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

# load data
with open("data/traces.jsonl", "r") as f:
    traces = [json.loads(line) for line in f.readlines()]
df = pd.DataFrame(traces)

# Transform TOP_RFHS_BY_LAYER_HEAD to map layer -> [rfh_indices] for each model
layer_to_top_rfhs = {}
for k, v in TOP_RFHS_BY_LAYER_HEAD.items():
    d_tmp = {}
    for cord in v:
        if cord[0] not in d_tmp:
            d_tmp[cord[0]] = []
        d_tmp[cord[0]].append(cord[1])
    layer_to_top_rfhs[k] = d_tmp

# instantiate patcher
patcher = OptimizedActivationPatcher(MODEL_ALIAS, device)

# filter dataframe and layer_rfh map for the specific model
df = df[df["model_name"] == MODEL_ALIAS].reset_index(drop=True)
layer_to_top_rfhs = layer_to_top_rfhs[MODEL_ALIAS]

# run patching
savedir = f"data/output/activation_patching/{MODEL_ALIAS}"
# this is for qwen-1.5B, we identify the topK RFH from the previous runs
topk_rfh_list = [(23, 2), (16, 2), (19, 1), (14, 3), (20, 9), (19, 5)]
os.makedirs(Path(savedir), exist_ok=True)
for i, row in df.iterrows():
    print(f"\n==========ID {i}: {row['unique_id']}==========\n")
    row = row.to_dict()
    patch_config = PatchConfig(
        all_rfh = False,
        layerwise_rfh = False,
        headwise_rfh = False,
        topk_rfh = True,
        topk_rfh_list = topk_rfh_list,
        all_rfh_savepath = f"{savedir}/all_rfh_heads.jsonl",
        layerwise_rfh_savepath = f"{savedir}/layer_<LAYER>_rfh_heads.jsonl",
        headwise_rfh_savepath = f"{savedir}/layer_<LAYER>_rfh_head_<HEAD>.jsonl",
        topk_rfh_savepath = f"{savedir}/top_<K>_rfh.jsonl"
    )
    try:
        patcher.run(
            response_withR = row["response_withR"],
            response_withoutR = row["response_withoutR"],
            heads_by_layer = layer_to_top_rfhs,
            config = patch_config
        )
    except RuntimeError as e:
        print(f"RuntimeError at index {i}, skipping. Error: {e}")
        clear_memory()
        continue
    