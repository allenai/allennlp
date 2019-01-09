import torch

from allennlp.common import Registrable


class Seq2SeqDecoder(torch.nn.Module, Registrable):
    # pylint: disable=abstract-method
    """
    A ``Seq2SeqDecoder`` is a ``Module`` that takes as input a result of ``Seq2SeqEncoder`` and returns a
    new sequence of vectors.

    Decoder should implement function to perform a single decoding step.
    """

    def __init__(self, output_dim):
        super(Seq2SeqDecoder, self).__init__()
        self._output_dim = output_dim

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this ``Seq2SeqDecoder``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise self.output_dim
