#!/usr/bin/env python3
"""Convert notebook logic for processing reasoning traces into a script."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
from typing import Dict, Iterable, List

import pandas as pd
from datasets import Dataset, load_dataset

MODEL_NAME_MAPPING: Dict[str, str] = {
    "DeepSeek-R1-Distill-Llama-8B": "llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B": "qwen-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "qwen-1p5B",
}

EOS = "<｜end▁of▁sentence｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_BLOCK_RE = r"<think>.*?</think>"
THINK_BLOCK_GROUPED_RE = r"(<think>)(.*?)(</think>)"
MASK = "\nOkay, I think I have finished thinking.\n"


def load_mmlu_datasets() -> Dict[str, Dataset]:
    """Download and cache the MMLU splits referenced by the trace IDs."""
    ds_physics = load_dataset("cais/mmlu", "college_physics", split="test")
    ds_math = load_dataset(
        "cais/mmlu",
        "college_mathematics",
        split="test",
    )
    return {
        "MMLU_physics": ds_physics,
        "MMLU": ds_math,
    }


def find_dataset_sample(sample_id: str, dataset_lookup: Dict[str, Dataset]):
    """Return the row dict for a sample ID like 'MMLU_physics_college_physics_0'."""
    for dataset_name in sorted(dataset_lookup.keys(), key=len, reverse=True):
        prefix = f"{dataset_name}_"
        if sample_id.startswith(prefix):
            tail = sample_id[len(prefix) :]
            _, idx_str = tail.rsplit("_", 1)
            idx = int(idx_str)
            return dataset_lookup[dataset_name][idx]
    raise ValueError(f"Unrecognized dataset prefix in sample_id: {sample_id}")


def get_problem(sample: dict) -> str:
    """Render the question plus its multiple-choice options."""
    question = sample["question"]
    choices = sample["choices"]
    choices_str = "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
    )
    return f"{question}\n{choices_str}"


def extract_boxed_answer(response: str) -> str | None:
    """Return the innermost value of the last \\boxed{} token, if present."""
    matches = re.findall(
        r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}",
        response,
    )
    if matches:
        return matches[-1]
    return None


def option_letter_to_index(letter: str) -> int:
    """Map answer letters A/B/C/D (case-insensitive) to 0/1/2/3."""
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    try:
        return mapping[letter.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported option letter: {letter}") from exc


def process_traces(
    traces: Iterable[dict],
    filename: str,
    dataset_lookup: Dict[str, Dataset],
) -> List[dict]:
    """Filter, sanitize, and augment traces for a single jsonl file."""
    traces = list(traces)
    results: List[dict] = []
    if "MMLU_physics" in filename:
        parts = filename.replace("MMLU_physics", "MMLU-physics").split("_")
    else:
        parts = filename.split("_")
    model_name = MODEL_NAME_MAPPING[parts[1]]

    for trace in traces:
        try:
            sample = find_dataset_sample(trace["unique_id"], dataset_lookup)
            gold_answer = sample["answer"]
            response = trace["response"].strip()

            if not response.endswith(EOS):
                continue
            if response.count(THINK_OPEN) != 1 or response.count(THINK_CLOSE) != 1:
                continue

            extracted_answer = extract_boxed_answer(response)
            if extracted_answer is None:
                continue

            extracted_idx = option_letter_to_index(extracted_answer)
            # print(
            #     "Extracted answer:",
            #     extracted_idx,
            #     ", gold answer:",
            #     gold_answer,
            # )
            if extracted_answer.lower() not in {"a", "b", "c", "d"}:
                continue

            assert 0 <= gold_answer <= 3
            is_correct = extracted_idx == gold_answer
            if not is_correct:
                continue

            match = re.search(THINK_BLOCK_RE, response, flags=re.DOTALL)
            if not match:
                continue

            res_with_r = (
                response[: match.end()]
                + r"The final answer is: \boxed{"
                + extracted_answer
                + "}"
            )
            res_without_r = re.sub(
                THINK_BLOCK_GROUPED_RE,
                r"\1" + MASK + r"\3",
                res_with_r,
                flags=re.DOTALL,
            )

            results.append(
                {
                    "model_name": model_name,
                    "unique_id": trace["unique_id"],
                    "problem": get_problem(sample),
                    "response_withR": res_with_r,
                    "response_withoutR": res_without_r,
                }
            )
        except Exception as exc:
            print(
                f"Error processing trace with unique_id {trace.get('unique_id')}: {exc}",
            )
            print(f"filename: {filename}")
            continue
    print(f"Processed {len(results)} traces out of {len(traces)} for {model_name}")
    return results


def read_trace_file(path: Path) -> List[dict]:
    """Load and parse every JSON line in a trace file."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process reasoning traces and export merged JSONL data.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(
            "/w/20252/wjcai/causal_inf/ACV/misc/R2A/output/reasoning_traces",
        ),
        help="Directory containing JSONL trace files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/traces.jsonl"),
        help="Path to write the final JSONL results.",
    )
    parser.add_argument(
        "--subject",
        choices=("math", "physics"),
        default="math",
        help="Which subject split to process (math=MMLU_*, physics=MMLU_physics_*).",
    )
    return parser.parse_args()


def main(trace_dir: Path, output_path: Path, subject: str) -> None:
    """Entry point for converting trace files into the consolidated dataset."""
    if not trace_dir.is_dir():
        raise ValueError(f"Trace directory does not exist: {trace_dir}")

    prefix_lookup = {
        "math": "MMLU_DeepSeek",
        "physics": "MMLU_physics_DeepSeek",
    }
    file_prefix = prefix_lookup[subject]

    dataset_lookup = load_mmlu_datasets()
    all_results: List[dict] = []

    for filename in os.listdir(trace_dir):
        if not filename.endswith(".jsonl"):
            continue
        if not filename.startswith(file_prefix):
            continue
        if "_withoutR_" in filename:
            continue
        file_path = trace_dir / filename
        traces = read_trace_file(file_path)
        file_results = process_traces(traces, filename, dataset_lookup)
        all_results.extend(file_results)

    if not all_results:
        print("No results found; nothing to write.")
        return

    df = pd.DataFrame(all_results)
    expected_models = len(set(MODEL_NAME_MAPPING.values()))
    valid_ids = (
        df.groupby("unique_id")["model_name"]
        .nunique()
        .pipe(lambda counts: counts[counts <= expected_models].index)
    )
    df_final = df[df["unique_id"].isin(valid_ids)].copy()
    if output_path.exists():
        print(f"Warning: Overwriting existing file at {output_path}")
        output_path.unlink()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_json(output_path, orient="records", lines=True)
    print(f"Wrote {len(df_final)} rows to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.trace_dir, args.output, args.subject)
