import torch

from allennlp.common import Params


class Seq2SeqEncoder(torch.nn.Module):
    """
    A ``Seq2SeqEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    modified sequence of vectors.  Input shape: ``(batch_size, sequence_length, input_dim)``; output
    shape: ``(batch_size, sequence_length, output_dim)``.

    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  You might need
    this if you want to construct a ``Linear`` layer using the output of this encoder, for
    instance.
    """
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the each vector in the sequence output by this ``Seq2SeqEncoder``.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        from allennlp.experiments.registry import Registry
        choice = params.pop_choice('type', Registry.list_seq2seq_encoders())
        return Registry.get_seq2seq_encoder(choice).from_params(params)
