from transformer_lens import HookedTransformer


def answer_map_to_tokens(model: HookedTransformer, answer_map: dict):
    answer_map_tokens = {
        key:model.to_tokens(value, prepend_bos=False).squeeze() for key, value in answer_map.items()
    }

    return answer_map_tokens