# allennlp.data.fields.sequence_label_field

## SequenceLabelField
```python
SequenceLabelField(self, labels:Union[List[str], List[int]], sequence_field:allennlp.data.fields.sequence_field.SequenceField, label_namespace:str='labels') -> None
```

A ``SequenceLabelField`` assigns a categorical label to each element in a
:class:`~allennlp.data.fields.sequence_field.SequenceField`.
Because it's a labeling of some other field, we take that field as input here, and we use it to
determine our padding and other things.

This field will get converted into a list of integer class ids, representing the correct class
for each element in the sequence.

Parameters
----------
labels : ``Union[List[str], List[int]]``
    A sequence of categorical labels, encoded as strings or integers.  These could be POS tags like [NN,
    JJ, ...], BIO tags like [B-PERS, I-PERS, O, O, ...], or any other categorical tag sequence. If the
    labels are encoded as integers, they will not be indexed using a vocab.
sequence_field : ``SequenceField``
    A field containing the sequence that this ``SequenceLabelField`` is labeling.  Most often, this is a
    ``TextField``, for tagging individual tokens in a sentence.
label_namespace : ``str``, optional (default='labels')
    The namespace to use for converting tag strings into integers.  We convert tag strings to
    integers for you, and this parameter tells the ``Vocabulary`` object which mapping from
    strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).

### count_vocab_items
```python
SequenceLabelField.count_vocab_items(self, counter:Dict[str, Dict[str, int]])
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
SequenceLabelField.index(self, vocab:allennlp.data.vocabulary.Vocabulary)
```

Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
This `modifies` the ``Field`` object, it does not return anything.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

### get_padding_lengths
```python
SequenceLabelField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### as_tensor
```python
SequenceLabelField.as_tensor(self, padding_lengths:Dict[str, int]) -> torch.Tensor
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
SequenceLabelField.empty_field(self) -> 'SequenceLabelField'
```

So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
option ``TextFields``), we need a representation of an empty field of each type.  This
returns that.  This will only ever be called when we're to the point of calling
:func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
``count_vocab_items``, etc., being called on this empty field.

We make this an instance method instead of a static method so that if there is any state
in the Field, we can copy it over (e.g., the token indexers in ``TextField``).

