import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides
from torch import nn
from torch.autograd import Variable

from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import (
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    SublayerConnection,
    subsequent_mask,
)
from allennlp.nn import util as nn_util


@DecoderNet.register("stacked_self_attention")
class StackedSelfAttentionDecoderNet(DecoderNet):
    """
    A Stacked self-attention decoder implementation.

    # Parameters

    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : `int`, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    use_positional_encoding : `bool`, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : `float`, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    residual_dropout_prob : `float`, optional, (default = 0.2)
        The dropout probability for the residual connections.
    attention_dropout_prob : `float`, optional, (default = 0.1)
        The dropout probability for the attention distributions in each attention layer.
    """

    def __init__(
        self,
        decoding_dim: int,
        target_embedding_dim: int,
        feedforward_hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        use_positional_encoding: bool = True,
        positional_encoding_max_steps: int = 5000,
        dropout_prob: float = 0.1,
        residual_dropout_prob: float = 0.2,
        attention_dropout_prob: float = 0.1,
    ) -> None:

        super().__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            decodes_parallel=True,
        )

        attn = MultiHeadedAttention(num_attention_heads, decoding_dim, attention_dropout_prob)
        feed_forward = PositionwiseFeedForward(decoding_dim, feedforward_hidden_dim, dropout_prob)
        self._embed_scale = math.sqrt(decoding_dim)
        self._positional_embedder = (
            PositionalEncoding(decoding_dim, positional_encoding_max_steps)
            if use_positional_encoding
            else None
        )
        self._dropout = nn.Dropout(dropout_prob)
        self._self_attention = Decoder(
            DecoderLayer(
                decoding_dim, deepcopy(attn), deepcopy(attn), feed_forward, residual_dropout_prob
            ),
            num_layers,
        )

    @overrides
    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        return {}

    @overrides
    def forward(
        self,
        previous_state: Dict[str, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
        previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        source_mask = source_mask.unsqueeze(-2)
        future_mask = Variable(
            subsequent_mask(previous_steps_predictions.size(-2), device=source_mask.device).type_as(
                source_mask.data
            )
        )
        if previous_steps_mask is None:
            previous_steps_mask = future_mask
        else:
            previous_steps_mask = previous_steps_mask.unsqueeze(-2) & future_mask
        previous_steps_predictions = previous_steps_predictions * self._embed_scale
        if self._positional_embedder:
            previous_steps_predictions = self._positional_embedder(previous_steps_predictions)
        previous_steps_predictions = self._dropout(previous_steps_predictions)
        decoded = self._self_attention(
            previous_steps_predictions, encoder_outputs, source_mask, previous_steps_mask
        )
        return {}, decoded


class Decoder(nn.Module):
    """
    Transformer N layer decoder with masking.
    Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn_util.clone(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    @overrides
    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    A single layer of transformer decoder.
    Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: F,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn_util.clone(SublayerConnection(size, dropout), 3)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        # Follow Figure 1 (right) for connections.
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayer[1](x, lambda y: self.src_attn(y, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)
