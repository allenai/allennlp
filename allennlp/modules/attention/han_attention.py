from typing import Optional, Tuple
from overrides import overrides
import torch
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import get_text_field_mask, masked_softmax


@Seq2VecEncoder.register("han_attention")
class HanAttention(Seq2VecEncoder):
    """
    Implements linear attention as described in
    https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427
    by Yang et. al, 2016.

    Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
    """
    def __init__(self,
                 input_dim: int = None,
                 context_vector_dim: int = None) -> None:
        super(HanAttention, self).__init__()
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

    @overrides
    def forward(self,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor):  # pylint: disable=arguments-differ
        batch_size, sequence_length, embedding_dim = matrix.size()
        attn_weights = matrix.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len
        attn_weights = masked_softmax(attn_weights, matrix_mask)
        attn_weights = (attn_weights
                        .unsqueeze(2)
                        .expand(batch_size, sequence_length, embedding_dim))
        return torch.sum(matrix * attn_weights, 1)
