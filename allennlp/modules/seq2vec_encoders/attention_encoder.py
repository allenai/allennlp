from overrides import overrides
import torch
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import masked_softmax

@Seq2VecEncoder.register("attention_encoder")
class AttentionEncoder(Seq2VecEncoder):
    """
    An ``AttentionEncoder`` is a :class:`Seq2VecEncoder` that computes a sentence-level representation
    via an attention computation over word sequence representations. As a :class:`Seq2VecEncoder`, the input to
    this module is of shape ``(batch_size, num_tokens, input_dim)``, and the output is of shape
    ``(batch_size, output_dim)``.

    This implementation is described in
    https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427
    by Yang et. al, 2016.

    As outlined in that paper, to compute a sentence representation, we first apply a linear transformation to each
    encoded word representation (projecting to ``context_vector_dim``), and then apply a non-linearity to each
    dimension of the resulting vectors.

    To get the attention weights, we then take the dot product of each of those word vectors with a learned
    context vector u_w (also of dimension ``context_vector_dim``), and subsequently normalize.

    Finally, we perform a weighted sum between each attention weight and its corresponding word vector.

    These operations result in sentence vectors that will have the same dimension as the input word
    representations.

    Parameters
    ----------
    input_dim : ``int``
        The dimension of the vector for each element in the input sequence;
        ``input_tensor.size(-1)``.
    context_vector_dim : ``int``
        We will do a linear projection of the input sequence to this dimension before
        performing the attention-weighted sum.
    activation : ``Activation``, optional (default=tanh)
        An activation function applied after the first linear projection.  Default is tanh
        activation.
    """
    def __init__(self,
                 input_dim: int = None,
                 context_vector_dim: int = None,
                 activation: Activation = None) -> None:
        super(AttentionEncoder, self).__init__()
        self._activation = activation or Activation.by_name('tanh')()
        self._mlp = torch.nn.Linear(input_dim, context_vector_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vector_dim,
                                                    1,
                                                    bias=False)
        self.vec_dim = self._mlp.weight.size(1)

    @overrides
    def get_input_dim(self) -> int:
        return self.vec_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.vec_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        batch_size, sequence_length, embedding_dim = tokens.size()
        tokens_ = tokens.view(batch_size * sequence_length, embedding_dim)
        attn_weights = self._activation(self._mlp(tokens_))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len

        attn_weights = masked_softmax(attn_weights, mask)
        attn_weights = (attn_weights
                        .unsqueeze(2)
                        .expand(batch_size, sequence_length, embedding_dim))

        attn_output = torch.sum(tokens * attn_weights, 1)

        return attn_output
