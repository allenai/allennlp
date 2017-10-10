from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

@Seq2VecEncoder.register("boe")
class BagOfEmbeddingsEncoder(Seq2VecEncoder):
    def __init__(self,
            embedding_dim: int,
            averaged: bool):
        super(BagOfEmbeddingsEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._averaged = averaged


    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  #pylint: disable=arguments-differ
        if tokens is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Out input is expected to have shape `(batch_size, num_tokens, embedding_dim)`, so we sum
        # out the `num_tokens` dimension.
        summed = tokens.sum(1)

        if self._averaged:
            lengths = get_lengths_from_binary_sequence_mask(mask)
            summed = summed / lengths.unsqueeze(-1).float()

        return summed

    @classmethod
    def from_params(cls, params: Params) -> 'BagOfEmbeddingsEncoder':
        embedding_dim = params.pop('embedding_dim')
        averaged = params.pop('averaged', default=None)
        return cls(embedding_dim=embedding_dim,
                   averaged=averaged)
