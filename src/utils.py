import transformer_lens.utils as utils
import plotly.express as px


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()