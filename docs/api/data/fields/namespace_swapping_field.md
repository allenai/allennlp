# allennlp.data.fields.namespace_swapping_field

## NamespaceSwappingField
```python
NamespaceSwappingField(self, source_tokens:List[allennlp.data.tokenizers.token.Token], target_namespace:str) -> None
```

A ``NamespaceSwappingField`` is used to map tokens in one namespace to tokens in another namespace.
It is used by seq2seq models with a copy mechanism that copies tokens from the source
sentence into the target sentence.

Parameters
----------
source_tokens : ``List[Token]``
    The tokens from the source sentence.
target_namespace : ``str``
    The namespace that the tokens from the source sentence will be mapped to.

### index
```python
NamespaceSwappingField.index(self, vocab:allennlp.data.vocabulary.Vocabulary)
```

Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
This `modifies` the ``Field`` object, it does not return anything.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

### get_padding_lengths
```python
NamespaceSwappingField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### as_tensor
```python
NamespaceSwappingField.as_tensor(self, padding_lengths:Dict[str, int]) -> torch.Tensor
```

Given a set of specified padding lengths, actually pad the data in this field and return a
torch Tensor (or a more complex data structure) of the correct shape.  We also take a
couple of parameters that are important when constructing torch Tensors.

Parameters
----------
padding_lengths : ``Dict[str, int]``
    This dictionary will have the same keys that were produced in
    :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
    relevant dimension, aggregated across all instances in a batch.

### empty_field
```python
NamespaceSwappingField.empty_field(self) -> 'NamespaceSwappingField'
```

So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
option ``TextFields``), we need a representation of an empty field of each type.  This
returns that.  This will only ever be called when we're to the point of calling
:func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
``count_vocab_items``, etc., being called on this empty field.

We make this an instance method instead of a static method so that if there is any state
in the Field, we can copy it over (e.g., the token indexers in ``TextField``).

