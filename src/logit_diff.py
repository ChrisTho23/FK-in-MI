from transformer_lens import HookedTransformer
import pandas as pd
import torch

def df_to_logits(model: HookedTransformer, df: pd.DataFrame):
  questions = list(df["question"])

  question_tokens = model.to_tokens(questions, padding_side='left', prepend_bos=True)

  logits, cache = model.run_with_cache(question_tokens)

  return logits, cache

def logits_to_logit_diff(df, logits, answer_map_tokens, device, per_prompt=False):
  final_logits = logits[:, -1, :]

  B, _ = final_logits.shape
  pos_tokens = answer_map_tokens["True"].expand(B, -1)
  neg_tokens = answer_map_tokens["False"].expand(B, -1)

  pos_logits = final_logits.gather(dim=-1, index=pos_tokens)
  neg_logits = final_logits.gather(dim=-1, index=neg_tokens)

  pos_logit_sum = pos_logits.sum(dim=-1)
  neg_logit_sum = neg_logits.sum(dim=-1)

  yes_mask = torch.tensor((df["answer"].values == "True"), dtype=torch.bool, device=device)

  logit_diff = torch.where(yes_mask, pos_logit_sum - neg_logit_sum, neg_logit_sum - pos_logit_sum)

  if per_prompt:
      return logit_diff
  else:
      return logit_diff.mean()

