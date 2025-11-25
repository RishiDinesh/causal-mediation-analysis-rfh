def get_layer_to_heads(top_heads):
    mapping = {}
    for k, v in top_heads.items():
        d_tmp = {}
        for cord in v:
            if cord[0] not in d_tmp:
                d_tmp[cord[0]] = []
            d_tmp[cord[0]].append(cord[1])
        mapping[k] = d_tmp
    return mapping

def merge_head_dicts(dict1, dict2):
    merged = {}
    for key in set(dict1) | set(dict2):
        merged[key] = []
        if key in dict1:
            merged[key].extend(dict1[key])
        if key in dict2:
            merged[key].extend(dict2[key])
    return merged