import gc
import json
import torch
from torch.nn.functional import cross_entropy
from transformer_lens import HookedTransformer
from src.suffix_map import get_suffix_map, SuffixMap
from src.constants import MODELS_LITERAL
from src._types import PatchConfig

torch.set_grad_enabled(False)

def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_json(data, fp):
    with open(fp, "a") as f:
        f.write(json.dumps(data) + "\n")

class ActivationPatcher:

    def __init__(self, model_alias: MODELS_LITERAL, device: str):
        model_to_path = {
            'llama-8B': r"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            'qwen-7B': r"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            'qwen-1p5B': r"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        }
        self.device = device
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_to_path[model_alias],
            dtype=torch.bfloat16,
            device = device
        )

    def _find_answer_start_idx(self, text: str) -> int:
        # Find first token after </think>, then skip "The final answer is:" preamble
        toks = self.model.to_str_tokens(text, prepend_bos=False)
        end_think_idx = toks.index("</think>")
        preamble_len = len(self.model.to_str_tokens("The final answer is:", prepend_bos=False))
        return end_think_idx + preamble_len
    
    def get_loss_and_log_probs(self, logits, tokens, idx):
        log_probs = torch.log_softmax(logits, dim=-1)
        answer_logits = logits[0, idx-1:-1, :]
        answer_tokens = tokens[0, idx:]
        answer_log_probs = log_probs[0, idx-1:-1, :][range(len(answer_tokens)), answer_tokens]
        loss = cross_entropy(answer_logits, answer_tokens, reduction='mean').item()
        joint_log_prob = answer_log_probs.sum().item()
        del log_probs, answer_logits, answer_tokens, answer_log_probs, 
        return loss, joint_log_prob
    
    def _collect_patched_metrics(
        self,
        tokens,
        patch_dict: dict[int, list[int]],
        z_cache,
        suffix_map: SuffixMap,
        start_idx: int,
        ablate: bool,
        prefix: str
    ) -> dict[str, float]:
        logits = self.patch_head_z(tokens, patch_dict, z_cache, suffix_map, ablate)
        loss, joint_log_prob = self.get_loss_and_log_probs(logits, tokens, start_idx)
        del logits
        clear_memory()
        return {
            f"{prefix}_loss": loss,
            f"{prefix}_joint_logprob": joint_log_prob
        }

    def _run_patching(
        self,
        base_metrics: dict[str, float],
        patch_dict: dict[int, list[int]],
        tok_withoutR,
        tok_withR,
        cache_withR,
        cache_withoutR,
        suffix_map: SuffixMap,
        start_idx_withoutR: int,
        start_idx_withR: int,
        save_path: str
    ) -> dict[str, float]:
        metrics = base_metrics.copy()
        metrics.update(self._collect_patched_metrics(
            tok_withoutR, patch_dict, cache_withR, suffix_map, start_idx_withoutR, False, "patched_withoutR"
        ))
        metrics.update(self._collect_patched_metrics(
            tok_withR, patch_dict, cache_withoutR, suffix_map, start_idx_withR, False, "patched_withR"
        ))
        metrics.update(self._collect_patched_metrics(
            tok_withR, patch_dict, cache_withoutR, suffix_map, start_idx_withR, True, "ablated_withR"
        ))
        save_json(
            data = metrics,
            fp = save_path
        )

    def run_with_cache_head_z(self, tokens, heads_by_layer):
        
        cache = {}
        def _cache(z, hook):
            layer = hook.layer()
            if layer not in heads_by_layer:
                return
            for h in heads_by_layer[layer]:
                cache[(layer, h)] = z[0, :, h, :].detach().to(device="cpu").contiguous() # [seq_long, d_head]
        
        name_filter = lambda n: n.endswith("attn.hook_z") or n.endswith(".hook_z")
        logits = self.model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[(
                name_filter,
                _cache
            )]
        )
        clear_memory()
        return logits, cache

    def patch_head_z(self, tokens, heads_by_layer, z_cache, suffix_map: SuffixMap, ablate: bool):
        
        idxs_short = suffix_map.idxs_short
        idxs_long  = suffix_map.idxs_long
        
        def _patch(z, hook):
            layer = hook.layer()
            if layer not in heads_by_layer:
                return z
            for h in heads_by_layer[layer]:
                cache = z_cache[(layer, h)].to(device=z.device, dtype=z.dtype) # [seq_long, d_head]
                if z.size(1) < cache.size(0):
                    z[0, idxs_short, h, :] = cache.index_select(0, idxs_long)
                else:
                    if ablate:
                        z[0, idxs_long, h, :] = 0 # patch the COT run, zero out head activations
                    else:
                        z[0, idxs_long, h, :] = cache.index_select(0, idxs_short)
            return z
        
        name_filter = lambda n: n.endswith("attn.hook_z") or n.endswith(".hook_z")
        logits = self.model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[(
                name_filter,
                _patch
            )]
        )
        return logits

    def run(
        self,
        response_withR: str,
        response_withoutR: str,
        heads_by_layer: dict[int, list[int]],
        config: PatchConfig,
        base_metrics: dict | None = None
    ):
        metrics = base_metrics if base_metrics is not None else {}
        with torch.inference_mode():
            # 1) tokenize
            tok_withR = self.model.to_tokens(response_withR, prepend_bos=False).to(device=self.device)
            tok_withoutR = self.model.to_tokens(response_withoutR, prepend_bos=False).to(device=self.device)

            # 2) build a suffix map (short ↔ long) after  comprehend
            suffix_map = get_suffix_map(tok_withoutR, tok_withR, self.model)

            # get start indices of the answer for loss and log prob calculations
            start_idx_withR = self._find_answer_start_idx(response_withR)
            start_idx_withoutR = self._find_answer_start_idx(response_withoutR)
            clear_memory()

            # 3) compute baseline losses and cache head-z for both
            withR_logits, cache_withR = self.run_with_cache_head_z(tok_withR, heads_by_layer)
            res = self.get_loss_and_log_probs(withR_logits, tok_withR, start_idx_withR)
            metrics.update({
                "withR_loss": res[0],
                "withR_joint_logprob": res[1]
            })
            del withR_logits
            clear_memory()

            withoutR_logits, cache_withoutR = self.run_with_cache_head_z(tok_withoutR, heads_by_layer)
            res = self.get_loss_and_log_probs(withoutR_logits, tok_withoutR, start_idx_withoutR)
            metrics.update({
                "withoutR_loss": res[0],
                "withoutR_joint_logprob": res[1]
            })
            del withoutR_logits
            clear_memory()

            def run_patch(patch_dict: dict[int, list[int]], save_path: str, message: str | None = None):
                if message:
                    print(message, flush=True)
                self._run_patching(
                    metrics,
                    patch_dict,
                    tok_withoutR,
                    tok_withR,
                    cache_withR,
                    cache_withoutR,
                    suffix_map,
                    start_idx_withoutR,
                    start_idx_withR,
                    save_path
                )

            if config.all_rfh:
                run_patch(
                    heads_by_layer,
                    config.all_rfh_savepath,
                    "Patching all given heads"
                )

            if config.layerwise_rfh:
                for layer in heads_by_layer:
                    patch_dict = {layer: heads_by_layer[layer]}
                    fp = config.layerwise_rfh_savepath.replace("<LAYER>", str(layer))
                    run_patch(patch_dict, fp, f"Patching all heads in layer {layer}")

            if config.headwise_rfh:
                for layer in heads_by_layer:
                    for head in heads_by_layer[layer]:
                        patch_dict = {layer: [head]}
                        fp = config.headwise_rfh_savepath.replace("<LAYER>", str(layer)).replace("<HEAD>", str(head))
                        run_patch(patch_dict, fp, f"Patching head index {head} in layer {layer}")
            
            if config.topk_rfh:
                for k in range(1, 6):
                    top_k = config.topk_rfh_list[:k]
                    patch_dict: dict[int, list[int]] = {}
                    for l, h in top_k:
                        if l not in patch_dict:
                            patch_dict[l] = []
                        patch_dict[l].append(h)
                    fp = config.topk_rfh_savepath.replace("<K>", str(k))
                    run_patch(patch_dict, fp, f"Patching top {k} heads: {top_k}")
            
            group_patch_runs = [
                (config.induction_heads, "Patching induction heads", config.induction_heads_savepath),
                (config.retrieval_heads, "Patching retrieval heads", config.retrieval_heads_savepath),
                (config.induction_and_retrieval_heads, "Patching induction and retrieval heads", config.induction_and_retrieval_heads_savepath),
                (config.induction_and_retrieval_and_rfh_heads, "Patching induction, retrieval and RFH heads", config.induction_and_retrieval_and_rfh_heads_savepath),
            ]

            for should_run, message, save_path in group_patch_runs:
                if should_run:
                    run_patch(heads_by_layer, save_path, message)
            del cache_withoutR
            del cache_withR
