from overrides import overrides

import torch
import torch.nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("cls_pooler")
class ClsPooler(Seq2VecEncoder):
    """
    Just takes the first vector from a list of vectors (which in a transformer is typically the
    [CLS] token) and returns it.

    # Parameters

    embedding_dim: int, optional
        This isn't needed for any computation that we do, but we sometimes rely on `get_input_dim`
        and `get_output_dim` to check parameter settings, or to instantiate final linear layers.  In
        order to give the right values there, we need to know the embedding dimension.  If you're
        using this with a transformer from the `transformers` library, this can often be found with
        `model.config.hidden_size`, if you're not sure.
    """

    def __init__(self, embedding_dim: int = None):
        super().__init__()
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        # tokens is assumed to have shape (batch_size, sequence_length, embedding_dim).  We just
        # want the first token for each instance in the batch.
        return tokens[:, 0]
