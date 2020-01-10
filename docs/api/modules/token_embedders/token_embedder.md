# allennlp.modules.token_embedders.token_embedder

## TokenEmbedder
```python
TokenEmbedder(self)
```

A ``TokenEmbedder`` is a ``Module`` that takes as input a tensor with integer ids that have
been output from a :class:`~allennlp.data.TokenIndexer` and outputs a vector per token in the
input.  The input typically has shape ``(batch_size, num_tokens)`` or ``(batch_size,
num_tokens, num_characters)``, and the output is of shape ``(batch_size, num_tokens,
output_dim)``.  The simplest ``TokenEmbedder`` is just an embedding layer, but for
character-level input, it could also be some kind of character encoder.

We add a single method to the basic ``Module`` API: :func:`get_output_dim()`.  This lets us
more easily compute output dimensions for the :class:`~allennlp.modules.TextFieldEmbedder`,
which we might need when defining model parameters such as LSTMs or linear layers, which need
to know their input dimension before the layers are called.

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
### get_output_dim
```python
TokenEmbedder.get_output_dim(self) -> int
```

Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
token.  This is `not` the shape of the returned tensor, but the last element of that shape.

