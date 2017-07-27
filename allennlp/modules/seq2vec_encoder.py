import torch

from allennlp.common import Params, Registrable


class Seq2VecEncoder(torch.nn.Module, Registrable):
    """
    A ``Seq2VecEncoder`` is a ``Module`` that takes as input a sequence of vectors and returns a
    single vector.  Input shape: ``(batch_size, sequence_length, input_dim)``; output shape:
    ``(batch_size, output_dim)``.

    We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  You might need
    this if you want to construct a ``Linear`` layer using the output of this encoder, for
    instance.
    """
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
