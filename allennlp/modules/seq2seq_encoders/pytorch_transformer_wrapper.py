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

    This class adapts the Transformer from torch.nn for use in AllenNLP.

    # Parameters

    input_dim : `int`, required.
        The input dimension of the encoder.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : `int`, required.
        The number of stacked self attention -> feedforward -> layer normalisation blocks.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    use_positional_encoding : `bool`, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : `float`, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    """  # noqa

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        num_attention_heads: int = 8,
        use_positional_encoding: bool = True,
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
        self._use_positional_encoding = use_positional_encoding
        self._input_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return True

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs

        output = self._transformer(output, mask)

        return output
