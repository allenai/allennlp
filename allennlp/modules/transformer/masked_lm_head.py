from typing import Union

import torch
from allennlp.common import FromParams
from allennlp.modules.transformer import TransformerModule
from transformers import PretrainedConfig
from transformers.activations import ACT2FN


class MaskedLMHead(TransformerModule, FromParams):
    _pretrained_mapping = {"LayerNorm": "layer_norm"}
    _pretrained_relevant_module = ["encoder", "bert.encoder", "roberta.encoder"]

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        activation: Union[str, torch.nn.Module] = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        if isinstance(activation, str):
            self.act_fn = ACT2FN[activation]
        else:
            self.act_fn = activation
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features: torch.Tensor):
        x = self.dense(features)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    @classmethod
    def _from_config(cls, config: PretrainedConfig, **kwargs) -> "MaskedLMHead":
        final_kwargs = {}
        final_kwargs["vocab_size"] = config.vocab_size
        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["layer_norm_eps"] = config.layer_norm_eps
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
