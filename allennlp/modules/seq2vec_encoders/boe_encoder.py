from overrides import overrides

import torch

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


@Seq2VecEncoder.register("boe")
@Seq2VecEncoder.register("bag_of_embeddings")
class BagOfEmbeddingsEncoder(Seq2VecEncoder):
    """
    A `BagOfEmbeddingsEncoder` is a simple [`Seq2VecEncoder`](./seq2vec_encoder.md) which simply sums
    the embeddings of a sequence across the time dimension. The input to this module is of shape
    `(batch_size, num_tokens, embedding_dim)`, and the output is of shape `(batch_size, embedding_dim)`.

    Registered as a `Seq2VecEncoder` with name "bag_of_embeddings" and "boe".

    # Parameters

    embedding_dim : `int`, required
        This is the input dimension to the encoder.
    averaged : `bool`, optional (default=`False`)
        If `True`, this module will average the embeddings across time, rather than simply summing
        (ie. we will divide the summed embeddings by the length of the sentence).
    """

    def __init__(self, embedding_dim: int, averaged: bool = False) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._averaged = averaged

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)

        # Our input has shape `(batch_size, num_tokens, embedding_dim)`, so we sum out the `num_tokens`
        # dimension.
        summed = tokens.sum(1)

        if self._averaged:
            if mask is not None:
                lengths = get_lengths_from_binary_sequence_mask(mask)
                length_mask = lengths > 0

                # Set any length 0 to 1, to avoid dividing by zero.
                lengths = torch.max(lengths, lengths.new_ones(1))
            else:
                lengths = tokens.new_full((1,), fill_value=tokens.size(1))
                length_mask = None

            summed = summed / lengths.unsqueeze(-1).float()

            if length_mask is not None:
                summed = summed * (length_mask > 0).unsqueeze(-1)

        return summed
