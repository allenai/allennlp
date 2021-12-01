from typing import Optional

from overrides import overrides
import torch
from torch import nn

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import add_positional_features


@Seq2SeqEncoder.register("pytorch_transformer")
class PytorchTransformer(Seq2SeqEncoder):
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in [Attention is all you Need]
    (https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077).

    This class adapts the Transformer from torch.nn for use in AllenNLP. Optionally, it adds positional encodings.

    Registered as a `Seq2SeqEncoder` with name "pytorch_transformer".

    # Parameters

    input_dim : `int`, required.
        The input dimension of the encoder.
    num_layers : `int`, required.
        The number of stacked self attention -> feedforward -> layer normalisation blocks.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    positional_encoding : `str`, optional, (default = `None`)
        Specifies the type of positional encodings to use. Your options are
         * `None` to have no positional encodings.
         * `"sinusoidal"` to have sinusoidal encodings, as described in https://api.semanticscholar.org/CorpusID:13756489.
         * `"embedding"` to treat positional encodings as learnable parameters
        Without positional encoding, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    positional_embedding_size : `int`, optional, (default = `512`)
        The number of positional embeddings.
    dropout_prob : `float`, optional, (default = `0.1`)
        The dropout probability for the feedforward network.
    activation : `str`, (default = `"relu"`)
        The activation function of intermediate layers. Must be either `"relu"` or `"gelu"`.
    """  # noqa

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        num_attention_heads: int = 8,
        positional_encoding: Optional[str] = None,
        positional_embedding_size: int = 512,
        dropout_prob: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout_prob,
            activation=activation,
        )
        self._transformer = nn.TransformerEncoder(layer, num_layers)
        self._input_dim = input_dim

        # initialize parameters
        # We do this before the embeddings are initialized so we get the default initialization for the embeddings.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if positional_encoding is None:
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding = True
            self._positional_embedding = None
        elif positional_encoding == "embedding":
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = nn.Embedding(positional_embedding_size, input_dim)
        else:
            raise ValueError(
                "positional_encoding must be one of None, 'sinusoidal', or 'embedding'"
            )

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        output = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids = torch.arange(inputs.size(1), dtype=torch.long, device=output.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        # For some reason the torch transformer expects the shape (sequence, batch, features), not the more
        # familiar (batch, sequence, features), so we have to fix it.
        output = output.permute(1, 0, 2)
        # For some other reason, the torch transformer takes the mask backwards.
        mask = ~mask
        output = self._transformer(output, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)

        return output
