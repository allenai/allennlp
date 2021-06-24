"""
An implementation of [T5](https://api.semanticscholar.org/CorpusID:204838007), adapted from [HuggingFace]
(https://github.com/huggingface/transformers/blob/4c32f9f26e6a84f0d9843fec8757e6ce640bb44e/src/transformers/models/t5/modeling_t5.py).
"""  # noqa: E401

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, TYPE_CHECKING

from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from allennlp.common import FromParams, Params, Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.attention_module import (
    T5Attention,
    AttentionOutput,
)
from allennlp.modules.transformer.util import (
    get_extended_attention_mask,
    FloatT,
    IntT,
    BoolT,
)
from allennlp.nn.beam_search import BeamSearch

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


class T5LayerNorm(TransformerModule, FromParams):
    """T5-style layer norm does not have bias and does not subtract the mean."""

    def __init__(self, hidden_size: int = 512, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states) -> FloatT:
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5FeedForwardProjection(TransformerModule, Registrable):
    def forward(self, hidden_states) -> FloatT:
        raise NotImplementedError


@T5FeedForwardProjection.register("relu")
class T5DenseReluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.wi = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size ** -0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states) -> FloatT:
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


@T5FeedForwardProjection.register("gated-gelu")
class T5DenseGatedGeluDense(TransformerModule, FromParams):
    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_0.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wi_1 = nn.Linear(hidden_size, ff_size, bias=False)
        self.wi_1.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.wo = nn.Linear(ff_size, hidden_size, bias=False)
        self.wo.weight.data.normal_(mean=0.0, std=ff_size ** -0.5)
        self.dropout = nn.Dropout(dropout)
        from allennlp.nn import Activation

        self.gelu_act = Activation.by_name("gelu_new")()

    def forward(self, hidden_states) -> FloatT:
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(TransformerModule, FromParams):
    _pretrained_mapping = {"DenseReluDense": "ff_proj"}

    def __init__(
        self,
        ff_proj: Optional[T5FeedForwardProjection] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff_proj = ff_proj or T5DenseReluDense()
        self.layer_norm = layer_norm or T5LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states) -> FloatT:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.ff_proj(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


@dataclass
class T5LayerSelfAttentionOutput:
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerSelfAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"SelfAttention": "self_attention"}

    def __init__(
        self,
        self_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        self.self_attention = self_attention or T5Attention(
            has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = layer_norm or T5LayerNorm(hidden_size=self.self_attention.hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def hidden_size(self) -> int:
        return self.self_attention.hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[FloatT]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5LayerSelfAttentionOutput:

        normed_hidden_states = self.layer_norm(hidden_states)

        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerSelfAttentionOutput(
            hidden_states,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


@dataclass
class T5LayerCrossAttentionOutput:
    hidden_states: FloatT
    attn_key_value_state: Optional[Tuple[FloatT, FloatT]]
    attn_position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5LayerCrossAttention(TransformerModule, FromParams):
    _pretrained_mapping = {"EncDecAttention": "enc_dec_attention"}

    def __init__(
        self,
        enc_dec_attention: Optional[T5Attention] = None,
        layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_dec_attention = enc_dec_attention or T5Attention(
            is_decoder=True,
            has_relative_attention_bias=False,
            is_cross_attention=True,
        )
        self.layer_norm = layer_norm or T5LayerNorm(hidden_size=self.enc_dec_attention.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: FloatT,
        key_value_states: Optional[FloatT],
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[Tuple[Tuple[FloatT]]] = None,
        use_cache: bool = False,
        query_length: int = None,
        output_attentions: bool = False,
    ) -> T5LayerCrossAttentionOutput:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output: AttentionOutput = self.enc_dec_attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output.hidden_states)

        return T5LayerCrossAttentionOutput(
            layer_output,
            attention_output.key_value_state,
            attention_output.position_bias,
            attention_output.attention_probs,
        )


KeyValueStates = Union[
    Tuple[FloatT, FloatT],  # without cross attention
    Tuple[FloatT, FloatT, FloatT, FloatT],  # with cross attention
]


@dataclass
class T5BlockOutput:
    hidden_states: FloatT
    present_key_value_states: Optional[KeyValueStates]
    self_attn_weights: Optional[FloatT]
    self_attn_position_bias: Optional[FloatT]
    cross_attn_weights: Optional[FloatT] = None
    cross_attn_position_bias: Optional[FloatT] = None


class T5Block(TransformerModule, FromParams):
    def __init__(
        self,
        attention: Optional[T5LayerSelfAttention] = None,
        cross_attention: Optional[T5LayerCrossAttention] = None,
        ff: Optional[T5LayerFF] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(attention or T5LayerSelfAttention())
        if cross_attention is None:
            self.is_decoder = False
        else:
            self.layer.append(cross_attention)
            self.is_decoder = True
        self.layer.append(ff or T5LayerFF())

    @property
    def hidden_size(self) -> int:
        return self.layer[0].hidden_size

    def forward(
        self,
        hidden_states: FloatT,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_bias: Optional[FloatT] = None,
        encoder_hidden_states: Optional[FloatT] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_decoder_position_bias: Optional[FloatT] = None,
        layer_head_mask: Optional[torch.BoolTensor] = None,
        encoder_layer_head_mask: Optional[torch.BoolTensor] = None,
        past_key_value: Optional[KeyValueStates] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5BlockOutput:
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = f"There should be {expected_num_past_key_values} past states. "
            error_message += "2 (past / key) for self attention. "
            if expected_num_past_key_values == 4:
                error_message += "2 (past / key) for cross attention. "
            error_message += f"Got {len(past_key_value)} past key / value states"
            assert len(past_key_value) == expected_num_past_key_values, error_message

        self_attention_outputs: T5LayerSelfAttentionOutput = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=None if past_key_value is None else past_key_value[:2],
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs.hidden_states
        present_key_value_state: Optional[
            Tuple[FloatT, FloatT]
        ] = self_attention_outputs.attn_key_value_state

        # clamp inf values to enable fp16 training
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs: T5LayerCrossAttentionOutput = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=None if past_key_value is None else past_key_value[2:],
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs.hidden_states
            if torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if (
                present_key_value_state is not None
                and cross_attention_outputs.attn_key_value_state is not None
            ):
                present_key_value_state: KeyValueStates = (  # type: ignore[no-redef]
                    present_key_value_state + cross_attention_outputs.attn_key_value_state
                )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        if torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        output = T5BlockOutput(
            hidden_states,
            present_key_value_state,
            self_attention_outputs.attn_weights,
            self_attention_outputs.attn_position_bias,
        )
        if do_cross_attention:
            output.cross_attn_weights = cross_attention_outputs.attn_weights
            output.cross_attn_position_bias = cross_attention_outputs.attn_position_bias
        return output


@dataclass
class T5StackOutput:
    last_hidden_state: FloatT
    past_key_values: Optional[List[KeyValueStates]] = None
    all_hidden_states: Optional[List[FloatT]] = None
    attentions: Optional[List[FloatT]] = None
    cross_attentions: Optional[List[FloatT]] = None


class T5Stack(TransformerModule, FromParams):
    _pretrained_mapping = {"embed_tokens": "token_embeddings", "block": "blocks"}

    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.is_decoder = blocks[0].is_decoder
        if not all(b.is_decoder == self.is_decoder for b in blocks):
            raise ConfigurationError("Found mismatched blocks in stack.")
        self.blocks = nn.ModuleList(blocks)
        self.token_embeddings = token_embeddings
        self.final_layer_norm = final_layer_norm or T5LayerNorm(hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def hidden_size(self) -> int:
        return self.blocks[0].hidden_size

    @staticmethod
    def get_head_mask(head_mask: Optional[torch.BoolTensor], num_hidden_layers: int) -> BoolT:
        if head_mask is not None:
            # -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.IntTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        encoder_hidden_states: Optional[FloatT] = None,
        encoder_attention_mask: Optional[torch.BoolTensor] = None,
        inputs_embeds: Optional[FloatT] = None,
        head_mask: Optional[torch.BoolTensor] = None,
        encoder_head_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[KeyValueStates] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_all_hidden_states: bool = False,
    ) -> T5StackOutput:
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs "
                f"and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.token_embeddings is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.token_embeddings(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            seq_length if past_key_values is None else past_key_values[0][0].shape[2] + seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, dtype=torch.bool, device=inputs_embeds.device
            )
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.bool
            )

        extended_attention_mask = get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.dtype, is_decoder=self.is_decoder
        )

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_blocks)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.num_blocks)
        present_key_value_states: Optional[List[KeyValueStates]] = [] if use_cache else None
        all_hidden_states: Optional[List[FloatT]] = [] if output_all_hidden_states else None
        all_attentions: Optional[List[FloatT]] = [] if output_attentions else None
        all_cross_attentions: Optional[List[FloatT]] = (
            [] if (output_attentions and self.is_decoder) else None
        )
        position_bias: Optional[FloatT] = None
        encoder_decoder_position_bias: Optional[FloatT] = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.blocks, past_key_values or [None] * self.num_blocks)
        ):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]
            if output_all_hidden_states:
                all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

            layer_outputs: T5BlockOutput = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs.hidden_states

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs.self_attn_position_bias
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs.cross_attn_position_bias
            if use_cache:
                present_key_value_states.append(layer_outputs.present_key_value_states)  # type: ignore
            if output_attentions:
                all_attentions.append(layer_outputs.self_attn_weights)  # type: ignore[union-attr]
                if self.is_decoder:
                    all_cross_attentions.append(layer_outputs.cross_attn_weights)  # type: ignore[union-attr]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_all_hidden_states:
            all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

        return T5StackOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            all_hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5EncoderStack(T5Stack, FromParams):
    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        if any(b.is_decoder for b in blocks):
            raise ConfigurationError("Found a decoder block in an encoder stack. This won't work.")

        super().__init__(
            token_embeddings,
            blocks,
            final_layer_norm=final_layer_norm,
            dropout=dropout,
        )

    @classmethod
    def basic_encoder(
        cls,
        token_embeddings: nn.Embedding,
        num_blocks: int = 6,
        block_self_attention: Lazy[T5Attention] = Lazy(T5Attention),
        final_layer_norm: Optional[T5LayerNorm] = None,
        block_ff: Lazy[T5LayerFF] = Lazy(T5LayerFF),
        dropout: float = 0.1,
    ) -> "T5EncoderStack":
        blocks = [
            T5Block(
                attention=T5LayerSelfAttention(
                    self_attention=block_self_attention.construct(
                        is_decoder=False, has_relative_attention_bias=(i == 0)
                    )
                ),
                cross_attention=None,
                ff=block_ff.construct(),
            )
            for i in range(num_blocks)
        ]
        return cls(token_embeddings, blocks, final_layer_norm=final_layer_norm, dropout=dropout)


class T5DecoderStack(T5Stack, FromParams):
    def __init__(
        self,
        token_embeddings: nn.Embedding,
        blocks: List[T5Block],
        final_layer_norm: Optional[T5LayerNorm] = None,
        dropout: float = 0.1,
    ):
        if not all(b.is_decoder for b in blocks):
            raise ConfigurationError("Found an encoder block in a decoder stack. This won't work.")

        super().__init__(
            token_embeddings,
            blocks,
            final_layer_norm=final_layer_norm,
            dropout=dropout,
        )

    @classmethod
    def basic_decoder(
        cls,
        token_embeddings: nn.Embedding,
        num_blocks: int = 6,
        block_self_attention: Lazy[T5Attention] = Lazy(T5Attention),
        block_cross_attention: Lazy[T5Attention] = Lazy(T5Attention),
        final_layer_norm: Optional[T5LayerNorm] = None,
        block_ff: Lazy[T5LayerFF] = Lazy(T5LayerFF),
        dropout: float = 0.1,
    ) -> "T5DecoderStack":
        blocks = [
            T5Block(
                attention=T5LayerSelfAttention(
                    self_attention=block_self_attention.construct(
                        is_decoder=True, has_relative_attention_bias=(i == 0)
                    )
                ),
                cross_attention=T5LayerCrossAttention(
                    enc_dec_attention=block_cross_attention.construct(
                        is_decoder=True,
                        has_relative_attention_bias=False,
                    )
                ),
                ff=block_ff.construct(),
            )
            for i in range(num_blocks)
        ]
        return cls(token_embeddings, blocks, final_layer_norm=final_layer_norm, dropout=dropout)


@dataclass
class T5Output:
    """
    Defines the output from the `T5` model.
    """

    encoder_last_hidden_state: FloatT
    """
    Final hidden states from the encoder.

    Shape: `(batch_size, target_length, hidden_dim)`
    """

    encoder_all_hidden_states: Optional[List[FloatT]] = None
    """
    All hidden states from the encoder.

    Shape (each): `(batch_size, target_length, hidden_dim)`
    """

    decoder_last_hidden_state: Optional[FloatT] = None
    """
    Final hidden states from the decoder. Only present when `labels` is given.

    Shape: `(batch_size, target_length, hidden_dim)`
    """

    decoder_all_hidden_states: Optional[List[FloatT]] = None
    """
    All hidden states from the decoder. Only present when `labels` is given
    and `output_all_hidden_states` is `True`.

    Shape (each): `(batch_size, target_length, hidden_dim)`
    """

    encoder_attentions: Optional[List[FloatT]] = None
    """
    Attention values from the encoder. Only present when `output_attentions` is `True`.
    """

    decoder_attentions: Optional[List[FloatT]] = None
    """
    Attention values from the decoder. Only present when `labels` is given
    and `output_attentions` is `True`.
    """

    cross_attentions: Optional[List[FloatT]] = None
    """
    Cross-attention values from the decoder. Only present when `labels` is given
    and `output_attentions` is `True`.
    """

    loss: Optional[FloatT] = None
    """
    The loss calculating with respect to `labels`.
    """

    logits: Optional[FloatT] = None
    """
    The logits that are used to calculate the loss with respect to `labels`.
    """

    predictions: Optional[IntT] = None
    """
    Predicted token IDs from beam search.

    Shape: `(batch_size, beam_size, max_decoding_steps)`.
    """

    predicted_log_probs: Optional[FloatT] = None
    """
    Probabilities corresponding to `predictions`.

    Shape: `(batch_size, beam_size,)`.
    """


class T5(TransformerModule, Registrable):

    _pretrained_mapping = {"shared": "token_embeddings"}

    # Don't know why HF has this param in their state_dict. It's not used in their model.
    _pretrained_ignore = [
        r"^decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight$"
    ]

    default_implementation = "default"

    def __init__(
        self,
        token_embeddings: Optional[nn.Embedding] = None,
        encoder: Lazy[T5EncoderStack] = Lazy(T5EncoderStack),
        decoder: Lazy[T5DecoderStack] = Lazy(T5DecoderStack),
        decoder_start_token_id: int = 0,
        pad_token_id: int = 0,  # These are both 0 in t5-(small|base|large). Go figure.
        eos_token_id: int = 1,
        vocab_size: int = 32128,
        model_dim: int = 512,
        output_attentions: bool = False,
        output_all_hidden_states: bool = False,
        beam_size: int = 3,
        max_decoding_steps: int = 100,
        tie_word_embeddings: bool = True,
    ):
        super().__init__()
        self._tie_word_embeddings = tie_word_embeddings

        self.model_dim = model_dim
        self.token_embeddings = token_embeddings or nn.Embedding(vocab_size, model_dim)
        if token_embeddings is None:
            self.token_embeddings.weight.data.normal_(mean=0.0, std=1.0)

        self.encoder: T5EncoderStack = encoder.construct(token_embeddings=self.token_embeddings)
        self.decoder: T5DecoderStack = decoder.construct(token_embeddings=self.token_embeddings)
        self.lm_head = nn.Linear(
            self.decoder.hidden_size, self.token_embeddings.num_embeddings, bias=False
        )
        if self._tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_attentions = output_attentions
        self.output_all_hidden_states = output_all_hidden_states

        self.beam_search = BeamSearch(
            self.eos_token_id, max_steps=max_decoding_steps, beam_size=beam_size or 1
        )

    @overrides
    def _post_load_pretrained_state_dict_hook(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> None:
        missing_keys_to_ignore = [
            "encoder.token_embeddings.weight",
            "decoder.token_embeddings.weight",
        ]
        if self._tie_word_embeddings:
            missing_keys_to_ignore.append("lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)

    @classmethod
    def _from_config(cls, config: "PretrainedConfig", **kwargs):
        attention_kwargs = {
            "hidden_size": config.d_model,
            "key_value_proj_dim": config.d_kv,
            "num_heads": config.num_heads,
            "relative_attention_num_buckets": config.relative_attention_num_buckets,
            "dropout": config.dropout_rate,
        }
        layer_norm_kwargs = {
            "hidden_size": config.d_model,
            "eps": config.layer_norm_epsilon,
        }
        block_ff = Lazy(
            T5LayerFF,
            params=Params(
                {
                    "ff_proj": {
                        "type": config.feed_forward_proj,
                        "hidden_size": config.d_model,
                        "ff_size": config.d_ff,
                        "dropout": config.dropout_rate,
                    },
                    "layer_norm": layer_norm_kwargs,
                    "dropout": config.dropout_rate,
                }
            ),
        )
        return cls(
            encoder=Lazy(
                T5EncoderStack.basic_encoder,
                contructor_extras={
                    "num_blocks": config.num_layers,
                    "block_self_attention": Lazy(T5Attention, contructor_extras=attention_kwargs),
                    "final_layer_norm": T5LayerNorm(**layer_norm_kwargs),
                    "block_ff": block_ff,
                    "dropout": config.dropout_rate,
                },
            ),
            decoder=Lazy(
                T5DecoderStack.basic_decoder,
                contructor_extras={
                    "num_blocks": config.num_decoder_layers,
                    "block_self_attention": Lazy(T5Attention, contructor_extras=attention_kwargs),
                    "block_cross_attention": Lazy(T5Attention, contructor_extras=attention_kwargs),
                    "final_layer_norm": T5LayerNorm(**layer_norm_kwargs),
                    "block_ff": block_ff,
                    "dropout": config.dropout_rate,
                },
            ),
            decoder_start_token_id=config.decoder_start_token_id,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            vocab_size=config.vocab_size,
            model_dim=config.d_model,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", config.tie_word_embeddings),
        )

    def _shift_right(self, input_ids, start_value: int):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = start_value

        return shifted_input_ids

    def _get_lm_logits(self, decoder_last_hidden_state: FloatT) -> FloatT:
        # Shape: (batch_size, target_length, model_dim)
        sequence_output = decoder_last_hidden_state
        # Rescale output before projecting on vocab
        # TODO: HF only does this when does this when embeddings are tied.
        # Currently tied embeddings is the only option we have, but if make
        # that configurable then we should put this in an 'if' block.
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        # Shape: (batch_size, target_length, vocab_size)
        logits = self.lm_head(sequence_output)
        return logits

    def forward(
        self,
        input_ids: IntT,
        attention_mask: Optional[BoolT] = None,
        labels: Optional[IntT] = None,
        decoder_attention_mask: Optional[BoolT] = None,
    ) -> T5Output:
        """
        Run forward pass of the model.
        """
        if attention_mask is None:
            attention_mask = ~(input_ids == self.pad_token_id)

        # Encode inputs.
        encoder_outputs: T5StackOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_all_hidden_states=self.output_all_hidden_states,
        )

        logits: Optional[FloatT] = None
        loss: Optional[FloatT] = None
        decoder_outputs: Optional[T5StackOutput] = None
        predictions: Optional[IntT] = None
        predicted_log_probs: Optional[FloatT] = None

        if labels is not None:
            # Calculate loss against targets.

            if decoder_attention_mask is None:
                decoder_attention_mask = ~(labels == self.pad_token_id)

            # Get decoder inputs from shifting lm labels to the right and pre-pending
            # the decoder start token ID.
            # Shape (both): (batch_size, target_length)
            decoder_input_ids = self._shift_right(labels, self.decoder_start_token_id)

            # Replace possible -100 values in labels by `pad_token_id`
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

            # Decode.
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                output_attentions=self.output_attentions,
                output_all_hidden_states=self.output_all_hidden_states,
            )

            # Shape: (batch_size, target_length, vocab_size)
            logits = self._get_lm_logits(decoder_outputs.last_hidden_state)  # type: ignore[union-attr]

            # Shape: (1,)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif self.training:
            raise ValueError("'labels' required during training")

        if not self.training:
            # Use beam search to generate a sequence of predicted tokens.

            # Shape: (batch_size, 1)
            initial_decoder_ids = torch.tensor(
                [[self.decoder_start_token_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            initial_state = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_outputs.last_hidden_state,
                "encoder_attention_mask": attention_mask,
            }

            # Run the beam search.
            # Shape (predictions): (batch_size, beam_size, max_decoding_steps)
            # Shape (predicted_log_probs):   (batch_size, beam_size)
            predictions, predicted_log_probs = self.beam_search.search(
                initial_decoder_ids, initial_state, self.take_search_step
            )

        return T5Output(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_all_hidden_states=encoder_outputs.all_hidden_states,
            decoder_last_hidden_state=(
                None if decoder_outputs is None else decoder_outputs.last_hidden_state
            ),
            decoder_all_hidden_states=(
                None if decoder_outputs is None else decoder_outputs.all_hidden_states
            ),
            encoder_attentions=encoder_outputs.attentions,
            decoder_attentions=None if decoder_outputs is None else decoder_outputs.attentions,
            cross_attentions=None if decoder_outputs is None else decoder_outputs.cross_attentions,
            loss=loss,
            logits=logits,
            predictions=predictions,
            predicted_log_probs=predicted_log_probs,
        )

    def take_search_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.
        """
        decoder_cache: Optional[List[KeyValueStates]] = None
        decoder_cache_dict = {
            k: state[k].contiguous() for k in state if k.startswith("decoder_cache_")
        }
        if decoder_cache_dict:
            decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)

        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        decoder_outputs: T5StackOutput = self.decoder(
            input_ids=last_predictions,
            past_key_values=decoder_cache,
            encoder_hidden_states=state["encoder_hidden_states"],
            encoder_attention_mask=state["encoder_attention_mask"],
            use_cache=True,
        )

        # Shape: (group_size, 2, vocab_size)
        lm_logits = self._get_lm_logits(decoder_outputs.last_hidden_state)

        # Shape: (group_size, vocab_size)
        logits = lm_logits[:, -1, :]

        # Shape: (group_size, vocab_size)
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Update state with decoder cache.
        decoder_cache = decoder_outputs.past_key_values
        assert decoder_cache is not None
        decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
        state.update(decoder_cache_dict)

        return log_probabilities, state

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: List[KeyValueStates]) -> Dict[str, torch.Tensor]:
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            # Each layer caches the key and value tensors for its self-attention and cross-attention.
            # Hence the `layer_cache` tuple has 4 elements.
            assert len(layer_cache) == 4
            for tensor_index, tensor in enumerate(layer_cache):
                key = f"decoder_cache_{layer_index}_{tensor_index}"
                cache_dict[key] = tensor
        return cache_dict

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> List[KeyValueStates]:
        decoder_cache: List[KeyValueStates] = []
        for block_index in range(self.decoder.num_blocks):
            base_key = f"decoder_cache_{block_index}_"
            layer_cache = (
                cache_dict[base_key + "0"].contiguous(),
                cache_dict[base_key + "1"].contiguous(),
                cache_dict[base_key + "2"].contiguous(),
                cache_dict[base_key + "3"].contiguous(),
            )
            decoder_cache.append(layer_cache)
        return decoder_cache


T5.register("default")(T5)
T5.register("from_pretrained", constructor="from_pretrained_module")(T5)
