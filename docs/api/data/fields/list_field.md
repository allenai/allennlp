# allennlp.data.fields.list_field

## ListField
```python
ListField(self, field_list:List[allennlp.data.fields.field.Field]) -> None
```

A ``ListField`` is a list of other fields.  You would use this to represent, e.g., a list of
answer options that are themselves ``TextFields``.

This field will get converted into a tensor that has one more mode than the items in the list.
If this is a list of ``TextFields`` that have shape (num_words, num_characters), this
``ListField`` will output a tensor of shape (num_sentences, num_words, num_characters).

Parameters
----------
field_list : ``List[Field]``
    A list of ``Field`` objects to be concatenated into a single input tensor.  All of the
    contained ``Field`` objects must be of the same type.

### count_vocab_items
```python
ListField.count_vocab_items(self, counter:Dict[str, Dict[str, int]])
```

If there are strings in this field that need to be converted into integers through a
:class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
of the vocabulary.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

A note on this ``counter``: because ``Fields`` can represent conceptually different things,
we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
mechanism to handle all mappings from strings to integers in all fields, while keeping
words in a ``TextField`` from sharing the same ids with labels in a ``LabelField`` (e.g.,
"entailment" or "contradiction" are labels in an entailment task)

Additionally, a single ``Field`` might want to use multiple namespaces - ``TextFields`` can
be represented as a combination of word ids and character ids, and you don't want words and
characters to share the same vocabulary - "a" as a word should get a different id from "a"
as a character, and the vocabulary sizes of words and characters are very different.

Because of this, the first key in the ``counter`` object is a `namespace`, like "tokens",
"token_characters", "tags", or "labels", and the second key is the actual vocabulary item.

### index
```python
ListField.index(self, vocab:allennlp.data.vocabulary.Vocabulary)
```

Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
This `modifies` the ``Field`` object, it does not return anything.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

### get_padding_lengths
```python
ListField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### sequence_length
```python
ListField.sequence_length(self) -> int
```

How many elements are there in this sequence?

### as_tensor
```python
ListField.as_tensor(self, padding_lengths:Dict[str, int]) -> ~DataArray
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

### batch_tensors
```python
ListField.batch_tensors(self, tensor_list:List[~DataArray]) -> ~DataArray
```

Takes the output of ``Field.as_tensor()`` from a list of ``Instances`` and merges it into
one batched tensor for this ``Field``.  The default implementation here in the base class
handles cases where ``as_tensor`` returns a single torch tensor per instance.  If your
subclass returns something other than this, you need to override this method.

This operation does not modify ``self``, but in some cases we need the information
contained in ``self`` in order to perform the batching, so this is an instance method, not
a class method.

