import plotly.express as px
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()

def answer_map_to_tokens(model: HookedTransformer, answer_map: dict):
    answer_map_tokens = {
        key:model.to_tokens(value, prepend_bos=False).squeeze() for key, value in answer_map.items()
    }

    return answer_map_tokens