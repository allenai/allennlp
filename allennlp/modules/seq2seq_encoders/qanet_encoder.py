from overrides import overrides
import torch
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import ModuleList
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.residual_with_layer_dropout import ResidualWithLayerDropout
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features
from allennlp.common.checks import check_dimensions_match


@Seq2SeqEncoder.register("qanet_encoder")
class QaNetEncoder(Seq2SeqEncoder):
    """
    Stack multiple QANetEncoderBlock into one sequence encoder.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for convolution output channels, multi-head attention output
        and the final output of feedforward layer.
    attention_projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_blocks : ``int``, required.
        The number of stacked encoder blocks.
    num_convs_per_block: ``int``, required.
        The number of convolutions in each block.
    conv_kernel_size: ``int``, required.
        The kernel size for convolution.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
        The initial dropout probability for layer dropout, and this might decay w.r.t the depth
        of the layer. For each mini-batch, the convolution/attention/ffn sublayer is
        stochastically dropped according to its layer dropout probability.
    attention_dropout_prob : ``float``, optional, (default = 0)
        The dropout probability for the attention distributions in the attention layer.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 attention_projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_blocks: int,
                 num_convs_per_block: int,
                 conv_kernel_size: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 layer_dropout_undecayed_prob: float = 0.1,
                 attention_dropout_prob: float = 0) -> None:
        super().__init__()

        self._input_projection_layer = None

        if input_dim != hidden_dim:
            self._input_projection_layer = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self._input_projection_layer = lambda x: x

        self._encoder_blocks = ModuleList([])
        for _ in range(num_blocks):
            encoder_block = QaNetEncoderBlock(hidden_dim,
                                              hidden_dim,
                                              attention_projection_dim,
                                              feedforward_hidden_dim,
                                              num_convs_per_block,
                                              conv_kernel_size,
                                              num_attention_heads,
                                              use_positional_encoding,
                                              dropout_prob,
                                              layer_dropout_undecayed_prob,
                                              attention_dropout_prob)
            self._encoder_blocks.append(encoder_block)

        self._input_dim = input_dim
        self._output_dim = hidden_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # pylint: disable=arguments-differ
        inputs = self._input_projection_layer(inputs)
        output = inputs
        for encoder_block in self._encoder_blocks:
            output = encoder_block(output, mask)
        return output


@Seq2SeqEncoder.register("qanet_encoder_block")
class QaNetEncoderBlock(Seq2SeqEncoder):
    """
    Implements the encoder block described in `QANet: Combining Local Convolution with Global
    Self-attention for Reading Comprehension <https://openreview.net/forum?id=B14TlG-RW>`_ .

    One encoder block mainly contains 4 parts:

        1. Add position embedding.
        2. Several depthwise seperable convolutions.
        3. Multi-headed self attention, which uses 2 learnt linear projections
           to perform a dot-product similarity between every pair of elements
           scaled by the square root of the sequence length.
        4. A two-layer FeedForward network.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for convolution output channels, multi-head attention output
        and the final output of feedforward layer.
    attention_projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_convs: ``int``, required.
        The number of convolutions in each block.
    conv_kernel_size: ``int``, required.
        The kernel size for convolution.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    layer_dropout_undecayed_prob : ``float``, optional, (default = 0.1)
        The initial dropout probability for layer dropout, and this might decay w.r.t the depth
        of the layer. For each mini-batch, the convolution/attention/ffn sublayer is randomly
        dropped according to its layer dropout probability.
    attention_dropout_prob : ``float``, optional, (default = 0)
        The dropout probability for the attention distributions in the attention layer.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 attention_projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_convs: int,
                 conv_kernel_size: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 layer_dropout_undecayed_prob: float = 0.1,
                 attention_dropout_prob: float = 0) -> None:
        super().__init__()

        check_dimensions_match(input_dim, hidden_dim, 'input_dim', 'hidden_dim')

        self._use_positional_encoding = use_positional_encoding

        self._conv_norm_layers = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_convs)])
        self._conv_layers = torch.nn.ModuleList()
        for _ in range(num_convs):
            padding = torch.nn.ConstantPad1d((conv_kernel_size // 2, (conv_kernel_size - 1) // 2), 0)
            depthwise_conv = torch.nn.Conv1d(hidden_dim, hidden_dim, conv_kernel_size, groups=hidden_dim)
            pointwise_conv = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
            self._conv_layers.append(
                    torch.nn.Sequential(padding, depthwise_conv, pointwise_conv, Activation.by_name("relu")())
            )

        self.attention_norm_layer = LayerNorm(hidden_dim)
        self.attention_layer = MultiHeadSelfAttention(num_heads=num_attention_heads,
                                                      input_dim=hidden_dim,
                                                      attention_dim=attention_projection_dim,
                                                      values_dim=attention_projection_dim,
                                                      attention_dropout_prob=attention_dropout_prob)
        self.feedforward_norm_layer = LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim,
                                       activations=[Activation.by_name('relu')(),
                                                    Activation.by_name('linear')()],
                                       hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                       num_layers=2,
                                       dropout=dropout_prob)

        self.dropout = Dropout(dropout_prob)
        self.residual_with_layer_dropout = ResidualWithLayerDropout(layer_dropout_undecayed_prob)
        self._input_dim = input_dim
        self._output_dim = hidden_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs

        total_sublayers = len(self._conv_layers) + 2
        sublayer_count = 0

        for conv_norm_layer, conv_layer in zip(self._conv_norm_layers, self._conv_layers):
            conv_norm_out = self.dropout(conv_norm_layer(output))
            conv_out = self.dropout(conv_layer(conv_norm_out.transpose_(1, 2)).transpose_(1, 2))
            sublayer_count += 1
            output = self.residual_with_layer_dropout(output, conv_out,
                                                      sublayer_count, total_sublayers)

        attention_norm_out = self.dropout(self.attention_norm_layer(output))
        attention_out = self.dropout(self.attention_layer(attention_norm_out, mask))
        sublayer_count += 1
        output = self.residual_with_layer_dropout(output, attention_out,
                                                  sublayer_count, total_sublayers)

        feedforward_norm_out = self.dropout(self.feedforward_norm_layer(output))
        feedforward_out = self.dropout(self.feedforward(feedforward_norm_out))
        sublayer_count += 1
        output = self.residual_with_layer_dropout(output, feedforward_out,
                                                  sublayer_count, total_sublayers)
        return output
