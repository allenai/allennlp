from typing import Optional, Dict
import torch

from allennlp.common import FromParams
from allennlp.modules.attention import Attention
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import apply_mask


class SelfAttention(TransformerModule, FromParams):
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
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

        self.scoring_func = scoring_func
        if self.scoring_func in ["additive", "linear", "bilinear"]:
            self.attn = Attention.by_name(self.scoring_func)(hidden_size, hidden_size)
        elif self.scoring_func == "scaled_dot_product":
            self.attn = Attention.by_name(self.scoring_func)(self.attention_head_size, False)
        else:
            self.attn = Attention.by_name(self.scoring_func)()

        # out linear layer for distilbert.
        if output_linear:
            self.output = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

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
        if key_states is None:
            key_states = query_states
        if value_states is None:
            value_states = query_states

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        attention_scores = self.attn(query_layer, key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = apply_mask(attention_scores, attention_mask)

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if hasattr(self, "output"):
            context_layer = self.output(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

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
