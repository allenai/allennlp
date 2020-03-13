"""
The BidirectionalTransformerEncoder from Calypso.
This is basically the transformer from https://nlp.seas.harvard.edu/2018/04/03/attention.html
so credit to them.

This code should be considered "private" in that we have several
transformer implementations and may end up deleting this one.
If you use it, consider yourself warned.
"""

from typing import Tuple, Callable
import math
import warnings

import torch
import torch.nn.functional as F

from allennlp.common.checks import ExperimentalFeatureWarning
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common import Registrable
from allennlp.nn import util


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.BoolTensor = None,
    dropout: Callable = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, util.min_value_of_dtype(scores.dtype))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size: int, device: str = "cpu") -> torch.BoolTensor:
    """Mask out subsequent positions."""
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool)).unsqueeze(0)
    return mask


class PositionalEncoding(torch.nn.Module, Registrable):
    """Implement the Positional Encoding function."""

    def __init__(self, input_dim: int, max_len: int = 5000) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, input_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.positional_encoding[:, : x.size(1)]


class PositionwiseFeedForward(torch.nn.Module):
    """Implements FFN equation."""

    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = torch.nn.Linear(input_dim, ff_dim)
        self.w_2 = torch.nn.Linear(ff_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(
        self, layer: torch.nn.Module, num_layers: int, return_all_layers: bool = False
    ) -> None:
        super().__init__()
        self.layers = util.clone(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self.return_all_layers = return_all_layers

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        all_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            if self.return_all_layers:
                all_layers.append(x)

        if self.return_all_layers:
            all_layers[-1] = self.norm(all_layers[-1])
            return all_layers
        return self.norm(x)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(
        self, size: int, self_attn: torch.nn.Module, feed_forward: torch.nn.Module, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = util.clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        # We assume d_v always equals d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        # These linear layers are
        #  [query_projection, key_projection, value_projection, concatenated_heads_projection]
        self.linears = util.clone(torch.nn.Linear(input_dim, input_dim), 4)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            # Shape (batch_size, num_heads, timesteps, timesteps)
            mask = mask.unsqueeze(1).expand([-1, self.num_heads, -1, -1])

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            layer(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


def make_model(
    num_layers: int = 6,
    input_size: int = 512,  # Attention size
    hidden_size: int = 2048,  # FF layer size
    heads: int = 8,
    dropout: float = 0.1,
    return_all_layers: bool = False,
) -> TransformerEncoder:
    """Helper: Construct a model from hyperparameters."""
    attn = MultiHeadedAttention(heads, input_size, dropout)
    ff = PositionwiseFeedForward(input_size, hidden_size, dropout)
    model = TransformerEncoder(
        EncoderLayer(input_size, attn, ff, dropout), num_layers, return_all_layers=return_all_layers
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


@Seq2SeqEncoder.register("bidirectional_language_model_transformer")
class BidirectionalLanguageModelTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        input_dropout: float = None,
        return_all_layers: bool = False,
    ) -> None:

        warnings.warn(
            "This particular transformer implementation is a provisional feature "
            "that's intended for AI2 internal use and might be deleted at any time. "
            "If you use it, consider yourself warned!",
            ExperimentalFeatureWarning,
        )

        super().__init__()

        self._return_all_layers = return_all_layers
        self.transformer_layers = num_layers
        self.num_layers = num_layers

        self._forward_transformer = make_model(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            return_all_layers=return_all_layers,
        )
        self._backward_transformer = make_model(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            return_all_layers=return_all_layers,
        )
        self._position = PositionalEncoding(input_dim)

        self.input_dim = input_dim
        self.output_dim = 2 * input_dim

        if input_dropout:
            self._dropout = torch.nn.Dropout(input_dropout)
        else:
            self._dropout = lambda x: x

        self.should_log_activations = False

    def get_attention_masks(self, mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns 2 masks of shape (batch_size, timesteps, timesteps) representing
        1) non-padded elements, and
        2) elements of the sequence which are permitted to be involved in attention at a given timestep.
        """
        device = mask.device
        # Forward case:
        timesteps = mask.size(1)
        # Shape (1, timesteps, timesteps)
        subsequent = subsequent_mask(timesteps, device)
        # Broadcasted logical and - we want zero
        # elements where either we have padding from the mask,
        # or we aren't allowed to use the timesteps.
        # Shape (batch_size, timesteps, timesteps)
        forward_mask = mask.unsqueeze(-1) & subsequent
        # Backward case - exactly the same, but transposed.
        backward_mask = forward_mask.transpose(1, 2)

        return forward_mask, backward_mask

    def forward(self, token_embeddings: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        forward_mask, backward_mask = self.get_attention_masks(mask)
        token_embeddings = self._position(token_embeddings)
        token_embeddings = self._dropout(token_embeddings)
        forward_output = self._forward_transformer(token_embeddings, forward_mask)
        backward_output = self._backward_transformer(token_embeddings, backward_mask)

        if self._return_all_layers:
            to_return = []
            for forward, backward in zip(forward_output, backward_output):
                to_return.append(torch.cat([forward, backward], -1))
            return to_return

        return torch.cat([forward_output, backward_output], -1)

    def get_regularization_penalty(self):
        return 0.0

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_bidirectional(self) -> bool:
        return True
