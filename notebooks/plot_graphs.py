import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

# ==========================================
# 1. CONSTANTS & CONFIGURATION
# ==========================================

SUBJECTS = ["math", "mmlu_math", "mmlu_physics"]
MODELS = ["qwen-1p5B", "qwen-7B"]

SUBJECT_TO_DATASET = {
    "math": "MATH-500",
    "mmlu_math": "MMLU-math",
    "mmlu_physics": "MMLU-physics"
}

MODEL_LABEL_MAP = {
    "qwen-1p5B": "Qwen-1.5B",
    "qwen-7B": "Qwen-7B",
}

DATA_DIR = Path("../data")
COMMON_IDS_PATH = DATA_DIR / "common_unique_ids.json"

# Plotting Configuration
def apply_plot_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

# ==========================================
# 2. DATA PROCESSING UTILITIES
# ==========================================

def get_common_ids_map() -> Dict[str, List[str]]:
    """Loads or generates the common IDs map."""
    # Logic from original script: Generate if needed, or just return existing
    # Note: For safety, I preserved the generation logic here.
    
    if COMMON_IDS_PATH.exists():
        with open(COMMON_IDS_PATH, "r") as f:
            return json.load(f)
            
    # If not exists, generate it (Section 1 of original code)
    print("Generating common unique IDs...")
    all_common = {}
    for subject in SUBJECTS:
        p1 = DATA_DIR / f"output_{subject}/activation_patching/qwen-1p5B/all_rfh_heads.jsonl"
        p2 = DATA_DIR / f"output_{subject}/activation_patching/qwen-7B/all_rfh_heads.jsonl"
        
        df1 = pd.read_json(p1, lines=True)
        df2 = pd.read_json(p2, lines=True)
        common = sorted(set(df1["unique_id"]) & set(df2["unique_id"]))
        all_common[subject] = common
        print(f"{subject}: {len(common)} common ids")

    with open(COMMON_IDS_PATH, "w") as f:
        json.dump(all_common, f, indent=2)
    return all_common


def load_and_filter_data(
    file_specs: List[Tuple[Dict[str, Any], str]], 
    common_ids: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Generic loader.
    file_specs: List of (metadata_dict, filepath)
    common_ids: Dictionary of subject -> list of valid IDs
    """
    dfs = []
    for metadata, path_str in file_specs:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Missing file {path}")
            continue
            
        try:
            temp = pd.read_json(path, lines=True)
            temp.drop_duplicates(subset="unique_id", inplace=True)
            
            # Filter by IDs if subject is provided in metadata
            if "subject" in metadata:
                subject = metadata["subject"]
                valid_ids = common_ids.get(subject, [])
                temp = temp[temp["unique_id"].isin(valid_ids)]
                # Map subject to display name immediately
                temp["dataset"] = SUBJECT_TO_DATASET.get(subject, subject)
            
            # Inject metadata columns
            for k, v in metadata.items():
                if k != "subject": # subject already handled via dataset map
                    temp[k] = v
            
            dfs.append(temp)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def aggregate_and_compute_metrics(
    df: pd.DataFrame, 
    group_cols: List[str]
) -> pd.DataFrame:
    """
    Centralized logic for TE, NIE, NDE computation.
    """
    # Columns of interest
    cols = ["withR_loss", "withoutR_loss", "patched_withoutR_loss", "patched_withR_loss"]
    if "ablated_withR_loss" in df.columns:
        cols.append("ablated_withR_loss")
        
    grouped = df.groupby(group_cols, as_index=False)[cols].mean()

    # Mapping
    # A: CoT=1, RFH=1 (Baseline)
    # B: CoT=0, RFH=0 (No CoT)
    # C: CoT=1, RFH=0 (Swap Target)
    # D: CoT=0, RFH=1 (NIE Target)
    # E: CoT=1, Ablated
    
    A = grouped["withR_loss"]
    B = grouped["withoutR_loss"]
    C = grouped["patched_withR_loss"]
    D = grouped["patched_withoutR_loss"]
    
    TE = B - A
    
    # Safe Division
    eps = 1e-8
    safe_TE = np.where(np.abs(TE) > eps, TE, np.nan)
    
    grouped["NIE_over_TE_pct"] = 100.0 * ((B - D) / safe_TE)
    grouped["NDE_swap_over_TE_pct"] = 100.0 * ((B - C) / safe_TE)
    
    if "ablated_withR_loss" in grouped.columns:
        E = grouped["ablated_withR_loss"]
        grouped["NDE_ablate_over_TE_pct"] = 100.0 * ((B - E) / safe_TE)

    return grouped

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_mediated_fraction_bars(df: pd.DataFrame, outfile: str):
    apply_plot_style()
    data = aggregate_and_compute_metrics(df, ["dataset", "model_name"])
    
    datasets = sorted(data["dataset"].unique())
    models = sorted(data["model_name"].unique())
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 2.9), sharey=True)
    if len(datasets) == 1: axes = [axes]
    
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(models))]
    width = 0.6
    
    global_vals = []
    
    for ax, ds in zip(axes, datasets):
        ds_data = data[data["dataset"] == ds]
        heights = []
        for m in models:
            val = ds_data.loc[ds_data["model_name"] == m, "NIE_over_TE_pct"]
            h = float(val.iloc[0]) if not val.empty else np.nan
            heights.append(h)
            if not np.isnan(h): global_vals.append(h)

        x = np.arange(len(models))
        ax.bar(x, heights, width=width, color=colors, edgecolor="black", linewidth=0.6)
        
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.9)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL_MAP.get(m, m) for m in models], rotation=0)
        ax.set_title(ds, loc="left")
        
        if ax is axes[0]: ax.set_ylabel("(NIE/TE) (%)")

    # Y Limits
    if global_vals:
        ymin = min(0.0, min(global_vals) - 5)
        ymax = max(0.0, max(global_vals) + 5)
        for ax in axes: ax.set_ylim(ymin, ymax)

    # Annotations
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axes):
        ax.text(0.02, 0.94, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    fig.text(0.5, 0.01, "Model", ha="center")
    
    # Dummy handles for legend
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors]
    labels = [MODEL_LABEL_MAP.get(m, m) for m in models]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=len(models), frameon=False)
    
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.92])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_layerwise_lines(df: pd.DataFrame, outfile: str):
    apply_plot_style()
    data = aggregate_and_compute_metrics(df, ["dataset", "layer", "model_name"])
    data["layer"] = data["layer"].astype(int)
    
    datasets = sorted(data["dataset"].unique())
    models = sorted(data["model_name"].unique())
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=(4.0, 2.6 * len(datasets)), sharex=True)
    if len(datasets) == 1: axes = [axes]
    
    cmap = plt.get_cmap("tab10")
    markers, linestyles = ["o", "^", "s"], ["-", "-", "-."]
    
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    model_handles = {}

    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        ds_data = data[data["dataset"] == ds]
        all_h = []
        
        for j, m in enumerate(models):
            sub = ds_data[ds_data["model_name"] == m].sort_values("layer")
            if sub.empty: continue
            
            y = sub["NIE_over_TE_pct"].values
            all_h.extend([v for v in y if not np.isnan(v)])
            
            (line,) = ax.plot(
                sub["layer"], y,
                marker=markers[j % len(markers)], linestyle=linestyles[j % len(linestyles)],
                color=cmap(j), linewidth=1.4, markersize=3.2,
                label=MODEL_LABEL_MAP.get(m, m)
            )
            model_handles[m] = line

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("NIE/TE (%)")
        ax.set_title(ds, loc="left")
        ax.text(0.01, 0.92, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        
        if all_h:
            ax.set_ylim(min(0.0, min(all_h)-5), max(0.0, max(all_h)+5))

    axes[-1].set_xlabel("Layer index")
    
    h_list = [model_handles[m] for m in models if m in model_handles]
    l_list = [h.get_label() for h in h_list]
    fig.legend(h_list, l_list, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(h_list), frameon=False)
    
    fig.tight_layout(rect=[0.08, 0.04, 0.98, 0.93])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_topk_grid(df: pd.DataFrame, outfile: str):
    apply_plot_style()
    data = aggregate_and_compute_metrics(df, ["dataset", "top_k", "model_name"])
    data["top_k"] = data["top_k"].astype(int)
    
    datasets = sorted(data["dataset"].unique())
    models = sorted(data["model_name"].unique())
    n_ds = len(datasets)
    
    fig, axes = plt.subplots(2, n_ds, figsize=(3.2 * n_ds, 4.6), sharex=True, sharey="row")
    if n_ds == 1: axes = np.array(axes).reshape(2, 1)
    
    cmap = plt.get_cmap("tab10")
    markers = ["o", "^", "s"]
    suff_handles, nec_handles = {}, {}
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    
    # Global limits
    suff_vals = data["NIE_over_TE_pct"].dropna().values
    nec_vals = np.concatenate([
        data["NDE_swap_over_TE_pct"].dropna().values, 
        data["NDE_ablate_over_TE_pct"].dropna().values
    ])
    
    ylims_s = (min(0, suff_vals.min()-5), max(0, suff_vals.max()+5)) if len(suff_vals) else (0,0)
    ylims_n = (min(0, nec_vals.min()-5), max(0, nec_vals.max()+5)) if len(nec_vals) else (0,0)

    for col, ds in enumerate(datasets):
        ds_data = data[data["dataset"] == ds]
        k_vals = sorted(ds_data["top_k"].unique())
        
        # Row 0: Sufficiency
        ax_s = axes[0, col]
        for j, m in enumerate(models):
            sub = ds_data[ds_data["model_name"] == m].sort_values("top_k")
            if sub.empty: continue
            
            c, mk = cmap(j), markers[j % len(markers)]
            (l,) = ax_s.plot(sub["top_k"], sub["NIE_over_TE_pct"], marker=mk, color=c, lw=1.4, ms=3.2)
            suff_handles[m] = l

        # Row 1: Necessity
        ax_n = axes[1, col]
        for j, m in enumerate(models):
            sub = ds_data[ds_data["model_name"] == m].sort_values("top_k")
            if sub.empty: continue
            
            c, mk = cmap(j), markers[j % len(markers)]
            (l_sw,) = ax_n.plot(sub["top_k"], sub["NDE_swap_over_TE_pct"], marker=mk, ls="-", color=c, lw=1.4, ms=3.0, label=f"{MODEL_LABEL_MAP[m]} (swap)")
            (l_ab,) = ax_n.plot(sub["top_k"], sub["NDE_ablate_over_TE_pct"], marker=mk, ls="--", color=c, lw=1.4, ms=3.0, label=f"{MODEL_LABEL_MAP[m]} (ablate)")
            
            if col == 0:
                nec_handles[f"{m}_swap"] = l_sw
                nec_handles[f"{m}_ablate"] = l_ab

        # Styling
        for ax in [ax_s, ax_n]:
            ax.axhline(0, color="k", lw=0.8, alpha=0.9)
            ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(k_vals)
        
        ax_s.set_title(ds)
        ax_s.set_ylim(ylims_s)
        ax_n.set_ylim(ylims_n)
        
        if col == 0:
            ax_s.set_ylabel("NIE/TE (%)")
            ax_n.set_ylabel("NDE/TE (%)")
        
        ax_s.text(0.02, 0.94, panel_labels[col*2], transform=ax_s.transAxes, fontweight="bold")
        ax_n.text(0.02, 0.94, panel_labels[col*2+1], transform=ax_n.transAxes, fontweight="bold")
        ax_n.set_xlabel("Top-k RFHs")

    # Legends
    fig.legend([suff_handles[m] for m in models if m in suff_handles], 
               [MODEL_LABEL_MAP[m] for m in models if m in suff_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=len(models), title="Models")
    
    fig.legend(list(nec_handles.values()), [h.get_label() for h in nec_handles.values()],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, title="Intervention")

    fig.tight_layout(rect=[0.06, 0.08, 0.98, 0.92])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_extended_ablation(df: pd.DataFrame, outfile: str):
    apply_plot_style()
    data = aggregate_and_compute_metrics(df, ["category", "model_name"])
    
    # Ordering
    cat_order = ["I", "IR", "IRR"]
    cat_labels = ["Induction heads", "Induction + retrieval", "Induction + retrieval + RFH"]
    data["category"] = pd.Categorical(data["category"], categories=cat_order, ordered=True)
    
    models = sorted(data["model_name"].unique())
    
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "^", "s"]
    handles = []

    vals = []
    
    for j, m in enumerate(models):
        sub = data[data["model_name"] == m].sort_values("category")
        if sub.empty: continue
        
        # Ensure x-axis alignment even if data missing
        sub = sub.set_index("category").reindex(cat_order)
        x = np.arange(len(cat_order))
        
        y_sw = sub["NDE_swap_over_TE_pct"].values
        y_ab = sub["NDE_ablate_over_TE_pct"].values
        vals.extend([v for v in np.concatenate([y_sw, y_ab]) if not np.isnan(v)])
        
        c, mk = cmap(j), markers[j % len(markers)]
        (l1,) = ax.plot(x, y_sw, marker=mk, ls="-", color=c, lw=1.4, ms=3.2, label=f"{MODEL_LABEL_MAP[m]} (swap)")
        (l2,) = ax.plot(x, y_ab, marker=mk, ls="--", color=c, lw=1.4, ms=3.2, label=f"{MODEL_LABEL_MAP[m]} (ablate)")
        handles.extend([l1, l2])

    ax.set_xticks(np.arange(len(cat_order)))
    ax.set_xticklabels(cat_labels, rotation=15, ha="right")
    ax.axhline(0, color="k", lw=0.8, alpha=0.9)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("NDE/TE (%)")
    ax.set_xlabel("Head configuration")
    
    if vals:
        ax.set_ylim(min(0, min(vals)-5), max(0, max(vals)+5))

    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4)
    fig.tight_layout(rect=[0.06, 0.05, 0.98, 0.90])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_pipeline():
    # 1. Setup Common IDs
    ids_map = get_common_ids_map()
    
    # 2. Metric 1: Global RFH
    print("--- Processing Global RFH ---")
    files_rfh = []
    for s in SUBJECTS:
        for m in MODELS:
            path = DATA_DIR / f"output_{s}/activation_patching/{m}/all_rfh_heads.jsonl"
            files_rfh.append(({"subject": s, "model_name": m}, str(path)))
            
    df_rfh = load_and_filter_data(files_rfh, ids_map)
    if not df_rfh.empty:
        plot_mediated_fraction_bars(df_rfh, "0_all_rfh_mediated_fraction.pdf")

    # 3. Metric 2: Layer-wise
    print("--- Processing Layer-wise ---")
    layer_map = {
        "qwen-1p5B": [1, 12, 14, 16, 19, 20, 23],
        "qwen-7B": [16, 19, 14, 1, 17, 22]
    }
    files_layer = []
    for s in SUBJECTS:
        for m in MODELS:
            for l in layer_map.get(m, []):
                path = DATA_DIR / f"output_{s}/activation_patching/{m}/layer_{l}_rfh_heads.jsonl"
                files_layer.append(({"subject": s, "model_name": m, "layer": l}, str(path)))
                
    df_layer = load_and_filter_data(files_layer, ids_map)
    if not df_layer.empty:
        plot_layerwise_lines(df_layer, "1_layerwise_rfh_mediated_fraction.pdf")

    # 4. Metric 3: Top-K
    print("--- Processing Top-K ---")
    files_topk = []
    for s in SUBJECTS:
        for m in MODELS:
            for k in range(1, 6):
                path = DATA_DIR / f"output_{s}/activation_patching/{m}/top_{k}_rfh.jsonl"
                files_topk.append(({"subject": s, "model_name": m, "top_k": k}, str(path)))
                
    df_topk = load_and_filter_data(files_topk, ids_map)
    if not df_topk.empty:
        plot_topk_grid(df_topk, "2_topk_rfh_suff_necess.pdf")

    # 5. Metric 4: Extended Circuit
    print("--- Processing Extended Circuit ---")
    circuit_files = [
        ("all_induction_heads", "I"), 
        ("all_induction_and_retrieval_heads", "IR"), 
        ("all_induction_retrieval_and_rfh_heads", "IRR")
    ]
    files_ext = []
    # Note: Original code hardcoded 'output_math' for this section, keeping consistent
    subject_fixed = "math" 
    for m in MODELS:
        for fname, cat_code in circuit_files:
            path = DATA_DIR / f"output_{subject_fixed}/activation_patching/{m}/{fname}.jsonl"
            files_ext.append(({"subject": subject_fixed, "model_name": m, "category": cat_code}, str(path)))
            
    df_ext = load_and_filter_data(files_ext, ids_map)
    if not df_ext.empty:
        plot_extended_ablation(df_ext, "3_extended_circuit_nde.pdf")

if __name__ == "__main__":
    run_pipeline()