import torch
import torch
from dataclasses import dataclass

@dataclass
class SuffixMap:
    """Mapping of token positions in short ↔ long after </think> tag."""
    idxs_short: torch.Tensor  
    idxs_long:  torch.Tensor

def get_suffix_map(tokens_short: torch.Tensor, tokens_long: torch.Tensor, model, close_tag="</think>") -> SuffixMap:
    """
    Get mapping of token positions in short ↔ long after </think> tag.
    """
    if tokens_short.dim() == 2: tokens_short = tokens_short[0]
    if tokens_long.dim()  == 2: tokens_long  = tokens_long[0]

    close_id = model.to_single_token(close_tag)

    i_s = tokens_short.numel() - 1
    i_l = tokens_long.numel()  - 1
    idxs_short, idxs_long = [], []

    while True:
        # Stop BEFORE including the close tag itself
        if tokens_short[i_s].item() == close_id:
            assert tokens_long[i_l].item() == close_id, "Close tag not aligned at same point."
            break
        # Map suffix positions 1:1
        idxs_short.append(i_s)
        idxs_long.append(i_l)

        i_s -= 1
        i_l -= 1
        assert i_s >= 0 and i_l >= 0, "Hit start before finding </think>."

    # We collected from end → start; flip to ascending order
    idxs_short = torch.tensor(list(reversed(idxs_short)), device=tokens_short.device, dtype=torch.long)
    idxs_long  = torch.tensor(list(reversed(idxs_long)),  device=tokens_long.device,  dtype=torch.long)
    return SuffixMap(idxs_short=idxs_short, idxs_long=idxs_long)