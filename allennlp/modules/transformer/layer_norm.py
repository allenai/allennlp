import torch

from allennlp.modules.transformer.transformer_module import TransformerModule


class LayerNorm(torch.nn.LayerNorm, TransformerModule):
    _huggingface_mapping = {"gamma": "weight", "beta": "bias"}
