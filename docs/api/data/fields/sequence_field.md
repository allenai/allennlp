# allennlp.data.fields.sequence_field

## SequenceField
```python
SequenceField(self, /, *args, **kwargs)
```

A ``SequenceField`` represents a sequence of things.  This class just adds a method onto
``Field``: :func:`sequence_length`.  It exists so that ``SequenceLabelField``, ``IndexField`` and other
similar ``Fields`` can have a single type to require, with a consistent API, whether they are
pointing to words in a ``TextField``, items in a ``ListField``, or something else.

### sequence_length
```python
SequenceField.sequence_length(self) -> int
```

How many elements are there in this sequence?

