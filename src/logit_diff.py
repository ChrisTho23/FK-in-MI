from typing import Dict, List

import pandas as pd
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer


def df_to_logits(model: HookedTransformer, df: pd.DataFrame):
  questions = list(df["question"])

  question_tokens = model.to_tokens(questions, padding_side='left', prepend_bos=True)

  logits, cache = model.run_with_cache(question_tokens)

  return logits, cache

def logits_to_logit_diff(
    df: pd.DataFrame, 
    logits: Float[torch.Tensor, "batch sequence d_model"], 
    answer_map_tokens: Dict[str, List[torch.Tensor]], 
    device: torch.device, 
    return_probs=False,
    per_prompt=False
):
    final_logits = logits[:, -1, :]
    assert len(answer_map_tokens["True"]) == len(answer_map_tokens["False"]), "must have equal number of true and false tokens"

    B, _ = final_logits.shape

    if return_probs:
        final_probs = final_logits.softmax(dim=-1)
    else:
        final_probs = final_logits

    pos_tokens = torch.tensor(answer_map_tokens["True"], device=device).clone().detach().expand(B, -1)
    neg_tokens = torch.tensor(answer_map_tokens["False"], device=device).clone().detach().expand(B, -1)

    pos_values = final_probs.gather(dim=-1, index=pos_tokens)
    neg_values = final_probs.gather(dim=-1, index=neg_tokens)

    if return_probs:
        pos_sum = pos_values.sum(dim=-1)
        neg_sum = neg_values.sum(dim=-1)
    else:
        pos_sum = pos_values.mean(dim=-1)
        neg_sum = neg_values.mean(dim=-1)

    yes_mask = torch.tensor(df["answer"].values, dtype=torch.bool, device=device)

    diff = torch.where(yes_mask, pos_sum - neg_sum, neg_sum - pos_sum)

    if per_prompt:
        return diff
    else:
        return diff.mean()

