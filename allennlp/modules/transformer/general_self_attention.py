import math
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from allennlp.common import FromParams
from allennlp.modules.attention import Attention
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import apply_mask


class GeneralSelfAttention(TransformerModule, FromParams):
    """
    TODO
    """

    def __init__(
        self,
        hidden_size: int = 512,
        attention_head_size: int = 64,
        num_attention_heads: int = 8,
        # has_relative_attention_bias: bool = False, # t5
        # relative_attention_num_buckets: int = 32, # t5
        # is_decoder: bool = False, # t5
        scoring_func: str = "scaled_dot_product",
        output_linear: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        normalize_weights: bool = False,
    ):

        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)

        if output_linear:
            self.output = torch.nn.Linear(hidden_size, self.all_head_size, bias=bias)

        self.scoring_func = scoring_func
        if self.scoring_func in ["additive", "linear", "bilinear"]:
            self.attn = Attention.by_name(self.scoring_func)(hidden_size, hidden_size)
        elif self.scoring_func == "scaled_dot_product":
            self.attn = Attention.by_name(self.scoring_func)(self.attention_head_size, False)
        else:
            self.attn = Attention.by_name(self.scoring_func)()

        # self.is_decoder = is_decoder
        # self.has_relative_attention_bias = has_relative_attention_bias
        # self.relative_attention_num_buckets = relative_attention_num_buckets

        # if self.has_relative_attention_bias:
        #     self.relative_attention_bias = torch.nn.Embedding(
        #         self.relative_attention_num_buckets, self.num_attention_heads
        #     )

        self.dropout = dropout

        if normalize_weights:
            self._normalize()

    def _normalize(self):
        self.query.weight.data.normal_(
            mean=0.0, std=(self.hidden_size * self.attention_head_size) ** -0.5
        )
        self.key.weight.data.normal_(mean=0.0, std=self.hidden_size ** -0.5)
        self.value.weight.data.normal_(mean=0.0, std=self.hidden_size ** -0.5)

        if hasattr(self, "output"):
            self.output.weight.data.normal_(
                mean=0.0, std=(self.num_attention_heads * self.attention_head_size) ** -0.5
            )

        # if self.has_relative_attention_bias:
        #     self.relative_attention_bias.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)

    def _transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _query_layer(self, query_states: torch.Tensor):
        mixed_query_layer = self.query(query_states)
        query_layer = self._transpose_for_scores(mixed_query_layer)
        return query_layer

    def _key_layer(self, key_states: torch.Tensor, past_key_states: Optional[torch.Tensor] = None):
        mixed_key_layer = self.key(key_states)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        return key_layer

    def _value_layer(
        self, value_states: torch.Tensor, past_value_states: Optional[torch.Tensor] = None
    ):
        mixed_value_layer = self.value(value_states)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        return value_layer

    def _get_attention_probs(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attention_scores = self.attn(query_layer, key_layer.transpose(-1, -2))

        # if position_bias is None:
        #     if self.has_relative_attention_bias:
        #         position_bias = self.compute_bias(real_seq_length, key_length)
        #     else:
        #         position_bias = torch.zeros(
        #             (1, self.num_attention_heads, real_seq_length, key_length),
        #             device=scores.device,
        #             dtype=scores.dtype,
        #         )

        #     # if key and values are already calculated
        #     # we want only the last query position bias
        #     if past_key_value is not None:
        #         position_bias = position_bias[:, :, -seq_length:, :]

        #     if mask is not None:
        #         # Shape: (batch_size, num_heads, seq_length, key_length)
        #         position_bias = apply_mask(position_bias, mask)

        # scores += position_bias

        if attention_mask is not None:
            attention_scores = apply_mask(attention_scores, attention_mask)

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        return attention_probs

    def _output_layer(self, attention_probs: torch.Tensor, value_layer: torch.Tensor):
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if hasattr(self, "output"):
            context_layer = self.output(context_layer)

        return context_layer

    def _get_key_value_states(
        self,
        query_states: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
    ):
        if key_states is None:
            key_states = query_states
        if value_states is None:
            value_states = query_states
        return key_states, value_states

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        query_states : `torch.Tensor`
            Shape `batch_size x seq_len x hidden_dim`
        key_states : `torch.Tensor`, optional
            Shape `batch_size x seq_len x hidden_dim`
        value_states : `torch.Tensor`, optional
            Shape `batch_size x seq_len x hidden_dim`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len`
        head_mask : `torch.BoolTensor`, optional
        output_attentions : `bool`
            Whether to also return the attention probabilities, default = `False`
        """
        # if key_states is None:
        #     key_states = query_states
        # if value_states is None:
        #     value_states = query_states

        key_states, value_states = self._get_key_value_states(
            query_states, key_states, value_states
        )

        query_layer = self._query_layer(query_states)
        key_layer = self._key_layer(key_states)
        value_layer = self._value_layer(value_states)

        attention_probs = self._get_attention_probs(
            query_layer, key_layer, attention_mask, head_mask
        )

        context_layer = self._output_layer(attention_probs, value_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Unfortunately mypy is insane, so we have to wrap these in unions.
FloatT = Union[torch.FloatTensor]
IntT = Union[torch.IntTensor]
BoolT = Union[torch.BoolTensor]


@dataclass
class T5AttentionOutput:
    hidden_states: FloatT
    key_value_state: Optional[Tuple[FloatT, FloatT]]
    position_bias: FloatT
    attn_weights: Optional[FloatT] = None


class T5Attention(GeneralSelfAttention):
    def __init__(
        self,
        is_decoder: bool = False,
        hidden_size: int = 512,
        key_value_proj_dim: int = 64,
        num_heads: int = 8,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        dropout: float = 0.1,
        normalize: bool = True,
    ):

        super().__init__(
            hidden_size=hidden_size,
            attention_head_size=key_value_proj_dim,
            num_attention_heads=num_heads,
            output_linear=True,
            dropout=dropout,
            bias=False,
            normalize_weights=normalize,
        )

        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets

        if self.has_relative_attention_bias:
            self.relative_attention_bias = torch.nn.Embedding(
                self.relative_attention_num_buckets, self.num_attention_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: IntT,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> IntT:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative positions are invalid. We use smaller
        buckets for small absolute relative_position and larger buckets for larger absolute relative_positions. All
        relative positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the
        same bucket. This should allow for more graceful generalization to longer sequences than the model has been
        trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range
            [0, num_buckets)
        """
        relative_buckets = relative_position.new_zeros(relative_position.shape)
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> FloatT:
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def _get_attention_probs(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        position_bias: torch.Tensor,
        real_seq_length: int,
        key_length: int,
        query_length: int,
    ):
        # compute scores
        scores = torch.matmul(
            query_layer, key_layer.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(real_seq_length, key_length)
            else:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )

            # if key and values are already calculated
            # we want only the last query position bias
            # TODO: use past_key_value correctly!!
            # if past_key_value is not None:
            #     position_bias = position_bias[:, :, -seq_length:, :]

            if attention_mask is not None:
                # Shape: (batch_size, num_heads, seq_length, key_length)
                position_bias = apply_mask(position_bias, attention_mask)

        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, num_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, num_heads, seq_length, key_length)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        return attn_weights

    def _get_key_value_states(
        self,
        query_states: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
    ):
        # TODO: simplify
        # FIX: past_key_value usage needs to be fixed.
        past_key_value = None
        if past_key_value is None:
            # if key_value_states is None:  # unnecessary check?
            key_value_states = (query_states, query_states)
        else:
            if key_value_states is None:
                # self-attn
                # (batch_size, num_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, query_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value

            key_value_states = (hidden_states, hidden_states)

        return key_value_states

    def _get_seq_key_length(self, hidden_states, past_key_value, key_value_states, query_length):
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        return real_seq_length, key_length

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        key_value_states: Optional[FloatT] = None,
        position_bias: Optional[FloatT] = None,
        past_key_value: Optional[Tuple[FloatT, FloatT]] = None,
        layer_head_mask: Optional[BoolT] = None,
        query_length: Optional[int] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> T5AttentionOutput:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by
        key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, num_heads, q_len - 1, dim_per_head)
        # batch_size, seq_length = hidden_states.shape[:2]
        # real_seq_length = seq_length

        real_seq_length, key_length = self._get_seq_key_length(
            hidden_states, past_key_value, key_value_states, query_length
        )
        # FIX: use key value states.
        key_value_states = self._get_key_value_states(hidden_states, None, None)

        # get query states
        query_states = self._query_layer(
            hidden_states
        )  # (batch_size, num_heads, seq_length, dim_per_head)

        key_states = self._key_layer(key_value_states[0])
        value_states = self._value_layer(key_value_states[1])

        attn_weights = self._get_attention_probs(
            query_states,
            key_states,
            mask,
            layer_head_mask,
            position_bias,
            real_seq_length,
            key_length,
            query_length,
        )

        attn_output = self._output_layer(attn_weights, value_states)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = T5AttentionOutput(attn_output, present_key_value_state, position_bias)
        if output_attentions:
            outputs.attn_weights = attn_weights
        return outputs


class SelfAttention(GeneralSelfAttention):
    """
    This module computes the self-attention, similar to the architecture in BERT. Additionally, the attention
    scoring function can be specified.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)

    # Parameters

    hidden_size: `int`
    num_attention_heads: `int`
    dropout: `float` (default = `0.0`)
    scoring_func: `str` (default = `scaled_dot_product`)
        The name of the attention-calculating function to be used.
        Eg. `additive`, `linear`, etc. For a complete list, please check :mod:`allennlp.modules.attention`.
    """

    _relevant_module = ["encoder.layers.0.attention.self", "encoder.layers.0.attention"]
    _huggingface_mapping = {"layer": "layers"}

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        scoring_func: str = "scaled_dot_product",
        output_linear: bool = False,
    ):

        attention_head_size = int(hidden_size / num_attention_heads)

        super().__init__(
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            num_attention_heads=num_attention_heads,
            scoring_func=scoring_func,
            output_linear=output_linear,
            dropout=dropout,
            bias=True,
        )

    @classmethod
    def _get_mapping(
        cls, pretrained_module=None, source="huggingface", mapping: Optional[Dict[str, str]] = None
    ):
        combined_mapping = {}
        if "huggingface" in source:
            combined_mapping.update(cls._huggingface_mapping)
        if mapping is not None:
            combined_mapping.update(mapping)
        if pretrained_module is not None:
            for name, _ in pretrained_module.named_modules():
                if "q_lin" in name:
                    combined_mapping["q_lin"] = "query"
                    combined_mapping["k_lin"] = "key"
                    combined_mapping["v_lin"] = "value"
                    combined_mapping["out_lin"] = "output"
                    combined_mapping["transformer"] = "encoder"
                    break
        return combined_mapping

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)
        final_kwargs = {}

        final_kwargs["hidden_size"] = submodules["query"].in_features
        if hasattr(submodules[""], "num_attention_heads"):
            final_kwargs["num_attention_heads"] = submodules[""].num_attention_heads
        elif hasattr(submodules[""], "n_heads"):
            final_kwargs["num_attention_heads"] = submodules[""].n_heads
            final_kwargs["output_linear"] = True  # Since this is the distilbert case.
        else:
            raise AttributeError("Cannot find a relevant attribute for number of heads.")

        final_kwargs["dropout"] = submodules["dropout"].p

        final_kwargs.update(**kwargs)

        return final_kwargs
