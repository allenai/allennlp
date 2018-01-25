
from overrides import overrides
import torch
from torch.nn import Linear

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.nn import util


@Seq2SeqEncoder.register("intra_sentence_attention")
class IntraSentenceAttentionEncoder(Seq2SeqEncoder):
    """
    An ``IntraSentenceAttentionEncoder`` is a :class:`Seq2SeqEncoder` that merges the original word
    representations with an attention (for each word) over other words in the sentence.  As a
    :class:`Seq2SeqEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, num_tokens, output_dim)``.

    We compute the attention using a configurable :class:`SimilarityFunction`, which could have
    multiple attention heads.  The operation for merging the original representations with the
    attended representations is also configurable (e.g., you can concatenate them, add them,
    multiply them, etc.).

    Parameters
    ----------
    input_dim : ``int``
        The dimension of the vector for each element in the input sequence;
        ``input_tensor.size(-1)``.
    projection_dim : ``int``, optional
        If given, we will do a linear projection of the input sequence to this dimension before
        performing the attention-weighted sum.
    similarity_function : ``SimilarityFunction``, optional
        The similarity function to use when computing attentions.  Default is to use a dot product.
    num_attention_heads: ``int``, optional
        If this is greater than one (default is 1), we will split the input into several "heads" to
        compute multi-headed weighted sums.  Must be used with a multi-headed similarity function,
        and you almost certainly want to do a projection in conjunction with the multiple heads.
    combination : ``str``, optional
        This string defines how we merge the original word representations with the result of the
        intra-sentence attention.  This will be passed to
        :func:`~allennlp.nn.util.combine_tensors`; see that function for more detail on exactly how
        this works, but some simple examples are ``"1,2"`` for concatenation (the default),
        ``"1+2"`` for adding the two, or ``"2"`` for only keeping the attention representation.
    output_dim : ``bool``, optional (default = None)
        The dimension of an optional output projection.
    """
    def __init__(self,
                 input_dim: int,
                 projection_dim: int = None,
                 similarity_function: SimilarityFunction = DotProductSimilarity(),
                 num_attention_heads: int = 1,
                 combination: str = '1,2',
                 output_dim: int = None) -> None:
        super(IntraSentenceAttentionEncoder, self).__init__()
        self._input_dim = input_dim
        if projection_dim:
            self._projection = torch.nn.Linear(input_dim, projection_dim)
        else:
            self._projection = lambda x: x
            projection_dim = input_dim
        self._matrix_attention = MatrixAttention(similarity_function)
        self._num_attention_heads = num_attention_heads
        if isinstance(similarity_function, MultiHeadedSimilarity):
            if num_attention_heads == 1:
                raise ConfigurationError("Similarity function has multiple heads but encoder doesn't")
            if num_attention_heads != similarity_function.num_heads:
                raise ConfigurationError("Number of heads don't match between similarity function "
                                         "and encoder: %d, %d" % (num_attention_heads,
                                                                  similarity_function.num_heads))
        elif num_attention_heads > 1:
            raise ConfigurationError("Encoder has multiple heads but similarity function doesn't")
        self._combination = combination

        combined_dim = util.get_combined_dim(combination, [input_dim, projection_dim])
        if output_dim:
            self._output_projection = Linear(combined_dim, output_dim)
            self._output_dim = output_dim
        else:
            self._output_projection = lambda x: x
            self._output_dim = combined_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        batch_size, sequence_length, _ = tokens.size()
        # Shape: (batch_size, sequence_length, sequence_length)
        similarity_matrix = self._matrix_attention(tokens, tokens)

        if self._num_attention_heads > 1:
            # In this case, the similarity matrix actually has shape
            # (batch_size, sequence_length, sequence_length, num_heads).  To make the rest of the
            # logic below easier, we'll permute this to
            # (batch_size, sequence_length, num_heads, sequence_length).
            similarity_matrix = similarity_matrix.permute(0, 1, 3, 2)

        # Shape: (batch_size, sequence_length, [num_heads,] sequence_length)
        intra_sentence_attention = util.last_dim_softmax(similarity_matrix.contiguous(), mask)

        # Shape: (batch_size, sequence_length, projection_dim)
        output_token_representation = self._projection(tokens)

        if self._num_attention_heads > 1:
            # We need to split and permute the output representation to be
            # (batch_size, num_heads, sequence_length, projection_dim / num_heads), so that we can
            # do a proper weighted sum with `intra_sentence_attention`.
            shape = list(output_token_representation.size())
            new_shape = shape[:-1] + [self._num_attention_heads, -1]
            # Shape: (batch_size, sequence_length, num_heads, projection_dim / num_heads)
            output_token_representation = output_token_representation.view(*new_shape)
            # Shape: (batch_size, num_heads, sequence_length, projection_dim / num_heads)
            output_token_representation = output_token_representation.permute(0, 2, 1, 3)

        # Shape: (batch_size, sequence_length, [num_heads,] projection_dim [/ num_heads])
        attended_sentence = util.weighted_sum(output_token_representation,
                                              intra_sentence_attention)

        if self._num_attention_heads > 1:
            # Here we concatenate the weighted representation for each head.  We'll accomplish this
            # just with a resize.
            # Shape: (batch_size, sequence_length, projection_dim)
            attended_sentence = attended_sentence.view(batch_size, sequence_length, -1)

        # Shape: (batch_size, sequence_length, combination_dim)
        combined_tensors = util.combine_tensors(self._combination, [tokens, attended_sentence])
        return self._output_projection(combined_tensors)

    @classmethod
    def from_params(cls, params: Params) -> 'IntraSentenceAttentionEncoder':
        input_dim = params.pop_int('input_dim')
        projection_dim = params.pop_int('projection_dim', None)
        similarity_function = SimilarityFunction.from_params(params.pop('similarity_function', {}))
        num_attention_heads = params.pop_int('num_attention_heads', 1)
        combination = params.pop('combination', '1,2')
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   projection_dim=projection_dim,
                   similarity_function=similarity_function,
                   num_attention_heads=num_attention_heads,
                   combination=combination)
