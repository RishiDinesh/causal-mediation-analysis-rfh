from dataclasses import dataclass

@dataclass
class PatchConfig:
    all_rfh: bool
    layerwise_rfh: bool
    headwise_rfh: bool
    topk_rfh: bool
    induction_heads: bool
    retrieval_heads: bool
    induction_and_retrieval_heads: bool
    induction_and_retrieval_and_rfh_heads: bool
    topk_rfh_list: list[tuple[int, int]]|None = None
    all_rfh_savepath: str|None = None
    layerwise_rfh_savepath: str|None = None
    headwise_rfh_savepath: str|None = None
    topk_rfh_savepath: str|None = None
    induction_heads_savepath: str|None = None
    retrieval_heads_savepath: str|None = None
    induction_and_retrieval_heads_savepath: str|None = None
    induction_and_retrieval_and_rfh_heads_savepath: str|None = None
    
    def __post_init__(self) -> None:
        missing = []
        if self.all_rfh and not self.all_rfh_savepath:
            missing.append("all_rfh_savepath")
        if self.layerwise_rfh and not self.layerwise_rfh_savepath:
            missing.append("layerwise_rfh_savepath")
        if self.headwise_rfh and not self.headwise_rfh_savepath:
            missing.append("headwise_rfh_savepath")
        if self.topk_rfh and (not self.topk_rfh_savepath or not self.topk_rfh_list):
            missing.append("topk_rfh_savepath")
        if self.induction_heads and not self.induction_heads_savepath:
            missing.append("induction_heads_savepath")
        if self.retrieval_heads and not self.retrieval_heads_savepath:
            missing.append("retrieval_heads_savepath")
        if self.induction_and_retrieval_heads and not self.induction_and_retrieval_heads_savepath:
            missing.append("induction_and_retrieval_heads_savepath")
        if self.induction_and_retrieval_and_rfh_heads and not self.induction_and_retrieval_and_rfh_heads_savepath:
            missing.append("induction_and_retrieval_and_rfh_savepath")
        if missing:
            raise ValueError(f"Savepath(s) required for enabled RFH: {', '.join(missing)}")
