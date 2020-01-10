# allennlp.nn.regularizers.regularizer

## Regularizer
```python
Regularizer(self, /, *args, **kwargs)
```

An abstract class representing a regularizer. It must implement
call, returning a scalar tensor.

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
