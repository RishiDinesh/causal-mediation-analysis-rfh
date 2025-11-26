import os
import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from src._types import PatchConfig
from src.patcher import ActivationPatcher, clear_memory
from src.utils import get_layer_to_heads, merge_head_dicts
from src.constants import (
    MODELS,
    MODELS_LITERAL,
    TOP_RFHS_BY_LAYER_HEAD,
    TOP_RETRIEVAL_HEADS,
    TOP_INDUCTION_HEADS
)

EXPERIMENT_CHOICES = [
    "all_rfh",
    "layerwise_rfh",
    "headwise_rfh",
    "topk_rfh",
    "induction_heads",
    "retrieval_heads",
    "induction_and_retrieval_heads",
    "induction_retrieval_and_rfh_heads"
]

parser = argparse.ArgumentParser(description="Run activation patching")
parser.add_argument("--model", type=str, choices=MODELS, help="Model", required=True)
parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_CHOICES, required=True, help="Experiments to run")
parser.add_argument("--n", type=int, default=1000, help="Number of samples to process")
args = parser.parse_args()

exclusive_group = ["induction_heads", "retrieval_heads", "induction_and_retrieval_heads", "induction_retrieval_and_rfh_heads"]
chosen_exclusive = [e for e in args.experiments if e in exclusive_group]

if len(chosen_exclusive) > 1:
    parser.error(
        "You may choose at most ONE of "
        f"{', '.join(sorted(exclusive_group))}. "
        f"Got: {chosen_exclusive}"
    )

print(f"Args: {args}", flush=True)

# set model
model_alias: MODELS_LITERAL = args.model

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}", flush=True)

# print gpu device details
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
    print(f"CUDA Version: {torch.version.cuda}", flush=True)
elif device.type == "mps":
    print("Using Apple Metal Performance Shaders (MPS)", flush=True)

# load data
with open("data/traces.jsonl", "r") as f:
    traces = [json.loads(line) for line in f.readlines()]
df = pd.DataFrame(traces)

# instantiate patcher
patcher = ActivationPatcher(model_alias, device)

if "induction_heads" in args.experiments:
    top_heads = TOP_INDUCTION_HEADS
elif "retrieval_heads" in args.experiments:
    top_heads = TOP_RETRIEVAL_HEADS
elif "induction_and_retrieval_heads" in args.experiments:
    top_heads = merge_head_dicts(TOP_INDUCTION_HEADS, TOP_RETRIEVAL_HEADS)
elif "induction_retrieval_and_rfh_heads" in args.experiments:
    temp = merge_head_dicts(TOP_INDUCTION_HEADS, TOP_RETRIEVAL_HEADS)
    top_heads = merge_head_dicts(temp, TOP_RFHS_BY_LAYER_HEAD)
elif any("rfh" in exp for exp in args.experiments):
    top_heads = TOP_RFHS_BY_LAYER_HEAD

# get layer -> head mapping
layer_to_heads = get_layer_to_heads(top_heads)

# filter dataframe and layer_rfh map for the specific model
df = df[df["model_name"] == model_alias].reset_index(drop=True)
layer_to_heads = layer_to_heads[model_alias]
print(f"Layer-to-head mapping for {model_alias}: {layer_to_heads}", flush=True)

# run patching
savedir = f"data/output/activation_patching/{model_alias}"
os.makedirs(Path(savedir), exist_ok=True)
print(f"Saving outputs to {savedir}", flush=True)

# we identify the topK RFH from the previous runs
topk_rfh_mapping = {
    "qwen-1p5B": [(23, 2), (16, 2), (19, 1), (14, 3), (20, 9), (19, 5)],
    "qwen-7B": [(22, 7), (19, 15), (16, 0), (17, 18), (14, 0), (16, 14)]
}
topk_rfh_list = topk_rfh_mapping[model_alias]

# create patching configuration
patch_config = PatchConfig(
    all_rfh = "all_rfh" in args.experiments,
    layerwise_rfh = "layerwise_rfh" in args.experiments,
    headwise_rfh = "headwise_rfh" in args.experiments,
    topk_rfh = "topk_rfh" in args.experiments,
    induction_heads = "induction_heads" in args.experiments,
    retrieval_heads = "retrieval_heads" in args.experiments,
    induction_and_retrieval_heads = "induction_and_retrieval_heads" in args.experiments,
    induction_and_retrieval_and_rfh_heads="induction_retrieval_and_rfh_heads" in args.experiments,
    topk_rfh_list = topk_rfh_list,
    all_rfh_savepath = f"{savedir}/all_rfh_heads.jsonl",
    layerwise_rfh_savepath = f"{savedir}/layer_<LAYER>_rfh_heads.jsonl",
    headwise_rfh_savepath = f"{savedir}/layer_<LAYER>_rfh_head_<HEAD>.jsonl",
    topk_rfh_savepath = f"{savedir}/top_<K>_rfh.jsonl",
    induction_heads_savepath = f"{savedir}/all_induction_heads.jsonl",
    retrieval_heads_savepath = f"{savedir}/all_retrieval_heads.jsonl",
    induction_and_retrieval_heads_savepath = f"{savedir}/all_induction_and_retrieval_heads.jsonl",
    induction_and_retrieval_and_rfh_heads_savepath=f"{savedir}/all_induction_retrieval_and_rfh_heads.jsonl",
)
print(f"Running with patch config: {patch_config}", flush=True)

# run experiments for each row
for i, row in df.iterrows():
    print(f"\n==========ID {i}: {row['unique_id']}==========\n", flush=True)
    row = row.to_dict()
    if i > args.n:
        break
    try:
        patcher.run(
            response_withR = row["response_withR"],
            response_withoutR = row["response_withoutR"],
            heads_by_layer = layer_to_heads,
            config = patch_config
        )
    except RuntimeError as e:
        print(f"RuntimeError at index {i}, skipping. Error: {e}", flush=True)
        clear_memory()
        continue

print("DONE!!", flush=True)