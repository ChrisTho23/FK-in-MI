import torch
from fancy_einsum import einsum
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer
import pandas as pd
from typing import List

def get_answer_residual_direction(
        model: HookedTransformer, 
        df: pd.DataFrame, 
        answer_map_tokens: dict[str, List[str]]
    ) -> torch.Tensor:
    W_U = model.W_U

    pos_unembedding = W_U[:, answer_map_tokens["Yes"]] # 2048,10
    neg_unembedding = W_U[:, answer_map_tokens["No"]] # 2048,3


    pos_unembedding_sum = pos_unembedding.sum(dim=-1).unsqueeze(0)
    neg_unembedding_sum = neg_unembedding.sum(dim=-1).unsqueeze(0)

    yes_mask = torch.tensor((df["answer"].values == "Yes"), dtype=torch.bool, device=device).unsqueeze(1)

    unembedding_diff = torch.where(yes_mask, pos_unembedding_sum - neg_unembedding_sum, neg_unembedding_sum - pos_unembedding_sum)

    return unembedding_diff

def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
) -> float:
    B = residual_stack.shape[1]

    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / B