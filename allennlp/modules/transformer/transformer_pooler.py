from typing import Dict, Optional, Any, Union, TYPE_CHECKING

import torch

from allennlp.common import FromParams
from allennlp.modules.transformer.activation_layer import ActivationLayer

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class TransformerPooler(ActivationLayer, FromParams):

    _pretrained_relevant_module = ["pooler", "bert.pooler", "roberta.pooler"]

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, torch.nn.Module] = "relu",
    ):
        super().__init__(hidden_size, intermediate_size, activation, pool=True)

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source: str = "huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        final_kwargs = {}

        final_kwargs["hidden_size"] = pretrained_module.dense.in_features
        final_kwargs["intermediate_size"] = pretrained_module.dense.out_features
        final_kwargs["activation"] = pretrained_module.activation

        final_kwargs.update(kwargs)

        return final_kwargs

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        return cls(config.hidden_size, config.hidden_size, "tanh")  # BERT has this hardcoded
