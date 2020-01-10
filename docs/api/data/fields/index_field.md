# allennlp.data.fields.index_field

## IndexField
```python
IndexField(self, index:int, sequence_field:allennlp.data.fields.sequence_field.SequenceField) -> None
```

An ``IndexField`` is an index into a
:class:`~allennlp.data.fields.sequence_field.SequenceField`, as might be used for representing
a correct answer option in a list, or a span begin and span end position in a passage, for
example.  Because it's an index into a :class:`SequenceField`, we take one of those as input
and use it to compute padding lengths.

Parameters
----------
index : ``int``
    The index of the answer in the :class:`SequenceField`.  This is typically the "correct
    answer" in some classification decision over the sequence, like where an answer span starts
    in SQuAD, or which answer option is correct in a multiple choice question.  A value of
    ``-1`` means there is no label, which can be used for padding or other purposes.
sequence_field : ``SequenceField``
    A field containing the sequence that this ``IndexField`` is a pointer into.

### get_padding_lengths
```python
IndexField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### as_tensor
```python
IndexField.as_tensor(self, padding_lengths:Dict[str, int]) -> torch.Tensor
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
IndexField.empty_field(self)
```

So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
option ``TextFields``), we need a representation of an empty field of each type.  This
returns that.  This will only ever be called when we're to the point of calling
:func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
``count_vocab_items``, etc., being called on this empty field.

We make this an instance method instead of a static method so that if there is any state
in the Field, we can copy it over (e.g., the token indexers in ``TextField``).

