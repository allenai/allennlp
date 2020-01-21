from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable


class Seq2SeqEncoder(_EncoderBase, Registrable):
    """
    A `Seq2SeqEncoder` is a `Module` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape : `(batch_size, sequence_length, input_dim)`; output
    shape : `(batch_size, sequence_length, output_dim)`.

    We add two methods to the basic `Module` API: `get_input_dim()` and `get_output_dim()`.
    You might need this if you want to construct a `Linear` layer using the output of this encoder,
    or to raise sensible errors for mis-matching input dimensions.
    """

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2SeqEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `Seq2SeqEncoder`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_bidirectional(self) -> bool:
        """
        Returns `True` if this encoder is bidirectional.  If so, we assume the forward direction
        of the encoder is the first half of the final dimension, and the backward direction is the
        second half.
        """
        raise NotImplementedError
