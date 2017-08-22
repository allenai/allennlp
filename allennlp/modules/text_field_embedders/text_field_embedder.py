from typing import Dict

import torch

from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary

class TextFieldEmbedder(torch.nn.Module, Registrable):
    """
    A ``TextFieldEmbedder`` is a ``Module`` that takes as input the
    :class:`~allennlp.data.DataArray` produced by a :class:`~allennlp.data.fields.TextField` and
    returns as output an embedded representation of the tokens in that field.

    The ``DataArrays`` produced by ``TextFields`` are `dictionaries` with named representations,
    like "words" and "characters".  When you create a ``TextField``, you pass in a dictionary of
    :class:`~allennlp.data.TokenIndexer` objects, telling the field how exactly the tokens in the
    field should be represented.  This class changes the type signature of ``Module.forward``,
    restricting ``TextFieldEmbedders`` to take inputs corresponding to a single ``TextField``,
    which is a dictionary of tensors with the same names as were passed to the ``TextField``.

    We also add a method to the basic ``Module`` API: :func:`get_output_dim()`.  You might need
    this if you want to construct a ``Linear`` layer using the output of this embedder, for
    instance.
    """
    default_implementation = 'basic'

    def forward(self,  # pylint: disable=arguments-differ
                text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the vector representing each token in the output of this
        ``TextFieldEmbedder``.  This is `not` the shape of the returned tensor, but the last element of
        that shape.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'TextFieldEmbedder':
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        return cls.by_name(choice).from_params(vocab, params)
