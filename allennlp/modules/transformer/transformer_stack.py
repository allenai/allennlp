from typing import Union, Optional, Tuple, TYPE_CHECKING
import logging
from dataclasses import dataclass

import torch

from allennlp.common import FromParams
from allennlp.modules.util import replicate_layers
from allennlp.modules.transformer.transformer_layer import TransformerLayer
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import FloatT

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)


@dataclass
class TransformerStackOutput:
    """
    Encapsulates the outputs of the `TransformerStack` module.
    """

    final_hidden_states: FloatT
    all_hidden_states: Optional[Tuple] = None
    all_self_attentions: Optional[Tuple] = None
    all_cross_attentions: Optional[Tuple] = None


class TransformerStack(TransformerModule, FromParams):
    """
    This module is the basic transformer stack.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    num_hidden_layers : `int`
    layer : `TransformerLayer`, optional
    hidden_size : `int`, optional
        This needs to be provided if no `layer` argument is passed.
    intermediate_size : `int`, optional
        This needs to be provided if no `layer` argument is passed.
    num_attention_heads : `int`
    attention_dropout : `float` (default = `0.0`)
        Dropout probability for the `SelfAttention` layer.
    hidden_dropout : `float` (default = `0.0`)
        Dropout probability for the `OutputLayer`.
    activation : `Union[str, torch.nn.Module]` (default = `"relu"`)
    add_cross_attention: `bool` (default = `False`)
        If True, the `TransformerLayer` modules will have cross attention modules as well.
        This is helpful when using the `TransformerStack` as a decoder.
    """

    _pretrained_mapping = {"layer": "layers"}
    _pretrained_relevant_module = ["encoder", "bert.encoder", "roberta.encoder"]

    def __init__(
        self,
        num_hidden_layers: int,
        layer: Optional[TransformerLayer] = None,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        activation: Union[str, torch.nn.Module] = "relu",
        add_cross_attention: bool = False,
    ):
        super().__init__()

        if layer is not None:
            logger.warning(
                "The `layer` argument has been specified. Any other arguments will be ignored."
            )
        else:
            assert (hidden_size is not None) and (intermediate_size is not None), "As the `layer`"
            "has not been provided, `hidden_size` and `intermediate_size` are"
            "required to create `TransformerLayer`s."

        layer = layer or TransformerLayer(
            hidden_size,  # type: ignore
            intermediate_size,  # type: ignore
            num_attention_heads,
            attention_dropout,
            hidden_dropout,
            activation,
            add_cross_attention,
        )
        self.layers = replicate_layers(layer, num_hidden_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> TransformerStackOutput:
        """
        # Parameters

        hidden_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        output_hidden_states : `bool`
            Whether to return the hidden_states for all layers, default = `False`
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self._add_cross_attention else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # type: ignore
                if self._add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # type: ignore

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        return TransformerStackOutput(
            hidden_states, all_hidden_states, all_attentions, all_cross_attentions
        )

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {}
        final_kwargs["num_hidden_layers"] = config.num_hidden_layers
        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["num_attention_heads"] = config.num_attention_heads
        final_kwargs["add_cross_attention"] = config.add_cross_attention
        final_kwargs["attention_dropout"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout"] = config.hidden_dropout_prob
        final_kwargs["intermediate_size"] = config.intermediate_size
        final_kwargs["activation"] = config.hidden_act
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
