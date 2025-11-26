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
        self.ablate = None

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
        mean_log_prob = answer_log_probs.mean().item()
        del log_probs, answer_logits, answer_tokens, answer_log_probs, 
        return loss, joint_log_prob, mean_log_prob
    
    def _collect_patched_metrics(
        self,
        tokens,
        patch_dict: dict[int, list[int]],
        z_cache,
        suffix_map: SuffixMap,
        start_idx: int,
        prefix: str
    ) -> dict[str, float]:
        logits = self.patch_head_z(tokens, patch_dict, z_cache, suffix_map)
        loss, joint_log_prob, mean_log_prob = self.get_loss_and_log_probs(logits, tokens, start_idx)
        del logits
        clear_memory()
        return {
            f"{prefix}_loss": loss,
            f"{prefix}_joint_logprob": joint_log_prob,
            f"{prefix}_mean_logprob": mean_log_prob
        }

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

    def patch_head_z(self, tokens, heads_by_layer, z_cache, suffix_map: SuffixMap):
        
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
                    if self.ablate:
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
        config: PatchConfig
    ):
        metrics, all_rfh_metrics, layerwise_rfh_metrics, headwise_rfh_metrics = {}, {}, {}, {}
        self.ablate = config.ablate
        print(f"Ablation: {self.ablate}", flush=True)
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
                "withR_joint_logprob": res[1],
                "withR_mean_logprob": res[2],
            })
            del withR_logits
            clear_memory()

            withoutR_logits, cache_withoutR = self.run_with_cache_head_z(tok_withoutR, heads_by_layer)
            res = self.get_loss_and_log_probs(withoutR_logits, tok_withoutR, start_idx_withoutR)
            metrics.update({
                "withoutR_loss": res[0],
                "withoutR_joint_logprob": res[1],
                "withoutR_mean_logprob": res[2]
            })
            del withoutR_logits
            clear_memory()

            if config.all_rfh:
                print("Patching all given heads", flush=True)
                all_rfh_metrics = metrics.copy()
                all_rfh_metrics.update(self._collect_patched_metrics(
                    tok_withoutR, heads_by_layer, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                ))
                all_rfh_metrics.update(self._collect_patched_metrics(
                    tok_withR, heads_by_layer, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                ))
                save_json(
                    data = all_rfh_metrics,
                    fp = config.all_rfh_savepath
                )

            if config.layerwise_rfh:
                for layer in heads_by_layer:
                    print(f"Patching all heads in layer {layer}", flush=True)
                    layerwise_rfh_metrics = metrics.copy()
                    patch_dict = {layer: heads_by_layer[layer]}
                    layerwise_rfh_metrics.update(self._collect_patched_metrics(
                        tok_withoutR, patch_dict, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                    ))
                    layerwise_rfh_metrics.update(self._collect_patched_metrics(
                        tok_withR, patch_dict, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                    ))
                    fp = config.layerwise_rfh_savepath.replace("<LAYER>", str(layer))
                    save_json(
                        data = layerwise_rfh_metrics,
                        fp = fp
                    )

            if config.headwise_rfh:
                for layer in heads_by_layer:
                    for head in heads_by_layer[layer]:
                        print(f"Patching head index {head} in layer {layer}", flush=True)
                        headwise_rfh_metrics = metrics.copy()
                        patch_dict = {layer: [head]}
                        headwise_rfh_metrics.update(self._collect_patched_metrics(
                            tok_withoutR, patch_dict, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                        ))
                        headwise_rfh_metrics.update(self._collect_patched_metrics(
                            tok_withR, patch_dict, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                        ))
                        fp = config.headwise_rfh_savepath.replace("<LAYER>", str(layer)).replace("<HEAD>", str(head))
                        save_json(
                            data = headwise_rfh_metrics,
                            fp = fp
                        )
            
            if config.topk_rfh:
                for k in range(1, 6):
                    topk_rfh_metrics = metrics.copy()
                    top_k = config.topk_rfh_list[:k]
                    print(f"Patching top {k} heads: {top_k}", flush=True)
                    patch_dict: dict[int, list[int]] = {}
                    for l, h in top_k:
                        if l not in patch_dict:
                            patch_dict[l] = []
                        patch_dict[l].append(h)
                    topk_rfh_metrics.update(self._collect_patched_metrics(
                        tok_withoutR, patch_dict, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                    ))
                    topk_rfh_metrics.update(self._collect_patched_metrics(
                        tok_withR, patch_dict, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                    ))
                    fp = config.topk_rfh_savepath.replace("<K>", str(k))
                    save_json(
                        data = topk_rfh_metrics,
                        fp = fp
                    )
            
            if config.induction_heads:
                print("Patching induction heads", flush=True)
                induction_head_metrics = metrics.copy()
                induction_head_metrics.update(self._collect_patched_metrics(
                    tok_withoutR, heads_by_layer, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                ))
                induction_head_metrics.update(self._collect_patched_metrics(
                    tok_withR, heads_by_layer, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                ))
                save_json(
                    data = induction_head_metrics,
                    fp = config.induction_heads_savepath
                )
            
            if config.retrieval_heads:
                print("Patching retrieval heads", flush=True)
                retrieval_head_metrics = metrics.copy()
                retrieval_head_metrics.update(self._collect_patched_metrics(
                    tok_withoutR, heads_by_layer, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                ))
                retrieval_head_metrics.update(self._collect_patched_metrics(
                    tok_withR, heads_by_layer, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                ))
                save_json(
                    data = retrieval_head_metrics,
                    fp = config.retrieval_heads_savepath
                )

            if config.induction_and_retrieval_heads:
                print("Patching induction and retrieval heads", flush=True)
                IR_metrics = metrics.copy()
                IR_metrics.update(self._collect_patched_metrics(
                    tok_withoutR, heads_by_layer, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                ))
                IR_metrics.update(self._collect_patched_metrics(
                    tok_withR, heads_by_layer, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                ))
                save_json(
                    data = IR_metrics,
                    fp = config.induction_and_retrieval_heads_savepath
                ) 

            if config.induction_and_retrieval_and_rfh_heads:
                print("Patching induction, retrieval and RFH heads", flush=True)
                IRR_metrics = metrics.copy()
                IRR_metrics.update(self._collect_patched_metrics(
                    tok_withoutR, heads_by_layer, cache_withR, suffix_map, start_idx_withoutR, "patched_withoutR"
                ))
                IRR_metrics.update(self._collect_patched_metrics(
                    tok_withR, heads_by_layer, cache_withoutR, suffix_map, start_idx_withR, "patched_withR"
                ))
                save_json(
                    data = IRR_metrics,
                    fp = config.induction_and_retrieval_and_rfh_heads_savepath
                )   
            del cache_withoutR
            del cache_withR
