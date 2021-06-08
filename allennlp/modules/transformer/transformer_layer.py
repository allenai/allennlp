from typing import Union, Optional, TYPE_CHECKING
from dataclasses import dataclass

import torch

from allennlp.common import FromParams
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.attention_module import SelfAttention, AttentionOutput
from allennlp.modules.transformer.output_layer import OutputLayer
from allennlp.modules.transformer.util import FloatT

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class AttentionLayer(TransformerModule, FromParams):
    """
    This module wraps the self-attention with the output-layer, similar to the architecture in BERT.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    hidden_size: `int`
    num_attention_heads: `int`
    attention_dropout: `float` (default = `0.0`)
        Dropout probability for the `SelfAttention` layer.
    hidden_dropout: `float` (default = `0.0`)
        Dropout probability for the `OutputLayer`.
    """

    _pretrained_relevant_module = "encoder.layer.0.attention"
    _pretrained_mapping = {"layer": "layers"}

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        is_cross_attention: bool = False,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.self = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout,
            is_cross_attention=is_cross_attention,
            is_decoder=is_decoder,
        )
        self.output = OutputLayer(hidden_size, hidden_size, hidden_dropout)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: bool = False,
    ):
        """
        # Parameters

        input_tensor : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        """

        if encoder_hidden_states is not None:
            attention_mask = encoder_attention_mask

        self_output = self.self(
            input_tensor,
            source_states=encoder_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        attention_output = self.output(self_output.hidden_states, input_tensor)
        outputs = AttentionOutput(
            attention_output,
            self_output.key_value_state,
            self_output.position_bias,
            self_output.attention_probs,
        )
        return outputs

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {}

        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["num_attention_heads"] = config.num_attention_heads
        final_kwargs["attention_dropout"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout"] = config.hidden_dropout_prob

        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)


@dataclass
class TransformerLayerOutput:
    """
    Encapsulates the outputs of the `TransformerLayer` module.
    """

    hidden_states: FloatT
    self_attention_probs: Optional[FloatT] = None
    cross_attention_probs: Optional[FloatT] = None


class TransformerLayer(TransformerModule, FromParams):
    """
    This module is a single transformer layer, mapping to `BertLayer` in the architecture in BERT.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    hidden_size : `int`
    intermediate_size : `int`
    num_attention_heads : `int`
    attention_dropout : `float` (default = `0.0`)
        Dropout probability for the `SelfAttention` layer.
    hidden_dropout : `float` (default = `0.0`)
        Dropout probability for the `OutputLayer`.
    activation : `Union[str, torch.nn.Module]`
    add_cross_attention : `bool` (default = `False`)
        If True, an extra `AttentionLayer` is added for cross-attention.
        This is helpful when using the layer in a decoder.
    """

    _pretrained_relevant_module = "encoder.layer.0"
    _pretrained_mapping = {
        "layer": "layers",
        "intermediate_act_fn": "act_fn",
        "crossattention": "cross_attention",
    }

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        activation: Union[str, torch.nn.Module] = "relu",
        add_cross_attention: bool = False,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self._add_cross_attention = add_cross_attention

        self.attention = AttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )

        if add_cross_attention:
            self.cross_attention = AttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                is_cross_attention=True,
                is_decoder=True,
            )

        self.intermediate = ActivationLayer(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output = OutputLayer(
            input_size=intermediate_size, hidden_size=hidden_size, dropout=hidden_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> TransformerLayerOutput:
        """
        # Parameters

        hidden_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        encoder_hidden_states : `torch.Tensor`, optional
        encoder_attention_mask : `torch.Tensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        """
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs.hidden_states
        self_attention_probs = attention_outputs.attention_probs
        cross_attention_probs = None

        if encoder_hidden_states is not None:
            assert hasattr(
                self, "cross_attention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated "
            "with cross-attention layers by setting `config.add_cross_attention=True`"

            cross_attention_outputs = self.cross_attention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs.hidden_states
            cross_attention_probs = cross_attention_outputs.attention_probs

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = TransformerLayerOutput(layer_output, self_attention_probs, cross_attention_probs)
        return outputs

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        final_kwargs = {}
        final_kwargs["hidden_size"] = config.hidden_size
        final_kwargs["num_attention_heads"] = config.num_attention_heads
        final_kwargs["attention_dropout"] = config.attention_probs_dropout_prob
        final_kwargs["hidden_dropout"] = config.hidden_dropout_prob
        final_kwargs["intermediate_size"] = config.intermediate_size
        final_kwargs["activation"] = config.hidden_act
        final_kwargs["add_cross_attention"] = config.add_cross_attention
        final_kwargs.update(**kwargs)
        return cls(**final_kwargs)
