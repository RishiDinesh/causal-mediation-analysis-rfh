import os
import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from src._types import PatchConfig
from src.constants import TOP_RFHS_BY_LAYER_HEAD, MODELS_LITERAL, MODELS
from src.optimized_patcher import OptimizedActivationPatcher, clear_memory

parser = argparse.ArgumentParser(description="Run activation patching")
parser.add_argument("--model", type=str, choices=MODELS, help="Model", required=True)
parser.add_argument("--experiments", nargs="+", choices=["all_rfh", "layerwise_rfh", "headwise_rfh", "topk_rfh"], required=True, help="Experiments to run")
parser.add_argument("--n", type=int, default=1000, help="Number of samples to process")
args = parser.parse_args()
print(f"Args: {args}")

model_alias: MODELS_LITERAL = args.model

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

# print gpu device details
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
elif device.type == "mps":
    print("Using Apple Metal Performance Shaders (MPS)")

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
patcher = OptimizedActivationPatcher(model_alias, device)

# filter dataframe and layer_rfh map for the specific model
df = df[df["model_name"] == model_alias].reset_index(drop=True)
layer_to_top_rfhs = layer_to_top_rfhs[model_alias]

# run patching
savedir = f"data/output/activation_patching/{model_alias}"
print(f"Saving outputs to {savedir}")

# we identify the topK RFH from the previous runs
topk_rfh_mapping = {
    "qwen-1p5B": [(23, 2), (16, 2), (19, 1), (14, 3), (20, 9), (19, 5)],
    "qwen-7B": [(22, 7), (19, 15), (16, 0), (17, 18), (14, 0), (16, 14)]
}
topk_rfh_list = topk_rfh_mapping[model_alias]

os.makedirs(Path(savedir), exist_ok=True)
for i, row in df.iterrows():
    print(f"\n==========ID {i}: {row['unique_id']}==========\n")
    row = row.to_dict()
    if i > args.n:
        break
    patch_config = PatchConfig(
        all_rfh = "all_rfh" in args.experiments,
        layerwise_rfh = "layerwise_rfh" in args.experiments,
        headwise_rfh = "headwise_rfh" in args.experiments,
        topk_rfh = "topk_rfh" in args.experiments,
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