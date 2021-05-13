import torch

from allennlp.modules.transformer.transformer_module import TransformerModule


class LayerNorm(torch.nn.LayerNorm, TransformerModule):
    _pretrained_mapping = {"gamma": "weight", "beta": "bias"}
