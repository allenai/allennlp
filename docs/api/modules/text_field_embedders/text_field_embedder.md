# allennlp.modules.text_field_embedders.text_field_embedder

## TextFieldEmbedder
```python
TextFieldEmbedder(self)
```

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

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### forward
```python
TextFieldEmbedder.forward(self, text_field_input:Dict[str, torch.Tensor], num_wrapping_dims:int=0, **kwargs) -> torch.Tensor
```

Parameters
----------
text_field_input : ``Dict[str, torch.Tensor]``
    A dictionary that was the output of a call to ``TextField.as_tensor``.  Each tensor in
    here is assumed to have a shape roughly similar to ``(batch_size, sequence_length)``
    (perhaps with an extra trailing dimension for the characters in each token).
num_wrapping_dims : ``int``, optional (default=0)
    If you have a ``ListField[TextField]`` that created the ``text_field_input``, you'll
    end up with tensors of shape ``(batch_size, wrapping_dim1, wrapping_dim2, ...,
    sequence_length)``.  This parameter tells us how many wrapping dimensions there are, so
    that we can correctly ``TimeDistribute`` the embedding of each named representation.

### get_output_dim
```python
TextFieldEmbedder.get_output_dim(self) -> int
```

Returns the dimension of the vector representing each token in the output of this
``TextFieldEmbedder``.  This is `not` the shape of the returned tensor, but the last element of
that shape.

