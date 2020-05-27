import torch

from allennlp.common import Registrable
from allennlp.data import TextFieldTensors


class TextFieldEmbedder(torch.nn.Module, Registrable):
    """
    A `TextFieldEmbedder` is a `Module` that takes as input the
    [`DataArray`](../../data/fields/text_field.md) produced by a [`TextField`](../../data/fields/text_field.md) and
    returns as output an embedded representation of the tokens in that field.

    The `DataArrays` produced by `TextFields` are _dictionaries_ with named representations, like
    "words" and "characters".  When you create a `TextField`, you pass in a dictionary of
    [`TokenIndexer`](../../data/token_indexers/token_indexer.md) objects, telling the field how exactly the
    tokens in the field should be represented.  This class changes the type signature of `Module.forward`,
    restricting `TextFieldEmbedders` to take inputs corresponding to a single `TextField`, which is
    a dictionary of tensors with the same names as were passed to the `TextField`.

    We also add a method to the basic `Module` API: `get_output_dim()`.  You might need this
    if you want to construct a `Linear` layer using the output of this embedder, for instance.
    """

    default_implementation = "basic"

    def forward(
        self, text_field_input: TextFieldTensors, num_wrapping_dims: int = 0, **kwargs
    ) -> torch.Tensor:
        """
        # Parameters

        text_field_input : `TextFieldTensors`
            A dictionary that was the output of a call to `TextField.as_tensor`.  Each tensor in
            here is assumed to have a shape roughly similar to `(batch_size, sequence_length)`
            (perhaps with an extra trailing dimension for the characters in each token).
        num_wrapping_dims : `int`, optional (default=`0`)
            If you have a `ListField[TextField]` that created the `text_field_input`, you'll
            end up with tensors of shape `(batch_size, wrapping_dim1, wrapping_dim2, ...,
            sequence_length)`.  This parameter tells us how many wrapping dimensions there are, so
            that we can correctly `TimeDistribute` the embedding of each named representation.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the vector representing each token in the output of this
        `TextFieldEmbedder`.  This is _not_ the shape of the returned tensor, but the last element
        of that shape.
        """
        raise NotImplementedError
