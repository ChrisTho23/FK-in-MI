import torch

def find_significant_locations(attn_head_out_act_patch_results, threshold=0.05):
    mask = (attn_head_out_act_patch_results > threshold) | (attn_head_out_act_patch_results < -threshold)

    indices = torch.nonzero(mask)

    layers = indices[:, 0] // 8
    heads = indices[:, 0] % 8
    positions = indices[:, 1]

    results = {
        f"L{layer}H{head}": round(attn_head_out_act_patch_results[layer * 8 + head, pos].item(), 2)
        for layer, head, pos in zip(layers, heads, positions)
    }

    sorted_results = dict(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True))

    return sorted_results