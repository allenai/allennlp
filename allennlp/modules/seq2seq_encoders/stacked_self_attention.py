from typing import List
from overrides import overrides
import torch
from torch.nn.modules import Dropout

from allennlp.common import Params
from allennlp.nn.util import add_positional_features
from allennlp.nn.activations import Activation
from allennlp.modules.similarity_functions import MultiHeadedSimilarity, SimilarityFunction, DotProductSimilarity
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders.intra_sentence_attention import IntraSentenceAttentionEncoder


@Seq2SeqEncoder.register("stacked_self_attention")
class StackedSelfAttentionEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    This encoder combines 3 layers in a 'block':
    1. A 2 layer FeedForward network.
    2. Multi-headed self attention, which uses 2 learnt linear projections
     to perform a dot-product similarity between every pair of elements
     scaled by the square root of the sequence length.
    3. Layer Normalisation.

    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for the self attention layers.
    projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    non_linear_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : ``int``, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    internal_similarity : ``SimilarityFunction``, optional (default = DotProductSimilarity(scale_output=True)).
        The internal similarity function to use for creating the attention distributions per
        attention head.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 non_linear_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 internal_similarity: SimilarityFunction = DotProductSimilarity(scale_output=True),
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.2) -> None:
        super(StackedSelfAttentionEncoder, self).__init__()

        self._use_positional_encoding = use_positional_encoding
        self._attention_layers: List[IntraSentenceAttentionEncoder] = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            # Project output of attention encoder through a feedforward network
            # and back to the input size for the next layer.
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[non_linear_hidden_dim, hidden_dim],
                                     num_layers=2)

            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            similarity_function = MultiHeadedSimilarity(num_attention_heads,
                                                        tensor_1_dim=hidden_dim,
                                                        internal_similarity=internal_similarity)
            self_attention = IntraSentenceAttentionEncoder(hidden_dim,
                                                           projection_dim,
                                                           similarity_function,
                                                           num_attention_heads,
                                                           combination='1,2',
                                                           output_projection_dim=hidden_dim)

            self.add_module(f"self_attention_{i}", self_attention)
            self._attention_layers.append(self_attention)

            layer_norm = LayerNorm(self_attention.get_output_dim())
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self._dropout = Dropout(p=dropout_prob)
        self._input_dim = input_dim
        self._output_dim = self._attention_layers[-1].get_output_dim()

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor): # pylint: disable=arguments-differ
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs
        for index, (attention, feedforward, layer_norm) in enumerate(zip(self._attention_layers,
                                                                         self._feedfoward_layers,
                                                                         self._layer_norm_layers)):
            cached_input = output
            # shape (batch_size, timesteps, input_size)
            non_linear_output = feedforward(output)
            if index != 0:
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                non_linear_output += cached_input
            # shape (batch_size, hidden_dim)
            attention_output = attention(non_linear_output, mask) + non_linear_output
            output = layer_norm(attention_output)
        return output

    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop('input_dim')
        hidden_dim = params.pop('hidden_dim')
        projection_dim = params.pop('projection_dim', None)
        non_linear_hidden_dim = params.pop("non_linear_hidden_dim")
        num_layers = params.pop("num_layers", 2)
        num_attention_heads = params.pop('num_attention_heads', 3)
        internal_similarity = SimilarityFunction.from_params(params.pop('internal_similarity', {}))
        use_positional_encoding = params.pop('use_positional_encoding', True)
        dropout_prob = params.pop("dropout_prob", 0.2)

        return cls(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   non_linear_hidden_dim=non_linear_hidden_dim,
                   projection_dim=projection_dim,
                   num_layers=num_layers,
                   num_attention_heads=num_attention_heads,
                   internal_similarity=internal_similarity,
                   use_positional_encoding=use_positional_encoding,
                   dropout_prob=dropout_prob)
