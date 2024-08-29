from typing import Dict, List

import pandas as pd
import torch
from fancy_einsum import einsum
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer


def get_answer_residual_direction(
        model: HookedTransformer, 
        df: pd.DataFrame, 
        answer_map_tokens: Dict[str, List[torch.Tensor]],
        device: torch.device
    ) -> Float[torch.Tensor, "batch d_model"]:
    W_U = model.W_U

    pos_unembedding = W_U[:, answer_map_tokens["True"]]
    neg_unembedding = W_U[:, answer_map_tokens["False"]] 

    pos_unembedding_sum = pos_unembedding.sum(dim=-1).unsqueeze(0)
    neg_unembedding_sum = neg_unembedding.sum(dim=-1).unsqueeze(0)

    yes_mask = torch.tensor(df["answer"].values, dtype=torch.bool, device=device).unsqueeze(1)

    unembedding_diff = torch.where(yes_mask, pos_unembedding_sum - neg_unembedding_sum, neg_unembedding_sum - pos_unembedding_sum)

    return unembedding_diff

def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    logit_diff_directions: Float[torch.Tensor, "batch d_model"],
    cache: ActivationCache,
    tokens_per_group: int,
) -> Float[torch.Tensor, "2 * n_layers - 1"]:
    B = residual_stack.shape[1]

    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / (B * tokens_per_group)

def increase_per_layer_type(logit_lens_logit_diffs: Float[torch.Tensor, "2 * n_layers - 1"]):
    increase_sa_layer = logit_lens_logit_diffs[3::2] - logit_lens_logit_diffs[2:-1:2]
    increase_mlp_layer = logit_lens_logit_diffs[4::2] - logit_lens_logit_diffs[3::2]

    sum_increase_sa_layer = increase_sa_layer.sum()
    sum_increase_mlp_layer = increase_mlp_layer.sum()

    print(
        f"Summed increase in self-attention layer after layer 1: {sum_increase_sa_layer:.2f}\n"
        f"Summed increase in MLP layer after layer 1: {sum_increase_mlp_layer:.2f}\n"
    )