# allennlp.data.fields.label_field

## LabelField
```python
LabelField(self, label:Union[str, int], label_namespace:str='labels', skip_indexing:bool=False) -> None
```

A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
text or 0-indexed integers (if you wish to skip indexing by passing skip_indexing=True).
If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
into integers.

This field will get converted into an integer index representing the class label.

Parameters
----------
label : ``Union[str, int]``
label_namespace : ``str``, optional (default="labels")
    The namespace to use for converting label strings into integers.  We map label strings to
    integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
    and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
    to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
    word).  If you have multiple different label fields in your data, you should make sure you
    use different namespaces for each one, always using the suffix "labels" (e.g.,
    "passage_labels" and "question_labels").
skip_indexing : ``bool``, optional (default=False)
    If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
    step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.

### count_vocab_items
```python
LabelField.count_vocab_items(self, counter:Dict[str, Dict[str, int]])
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
LabelField.index(self, vocab:allennlp.data.vocabulary.Vocabulary)
```

Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
This `modifies` the ``Field`` object, it does not return anything.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

### get_padding_lengths
```python
LabelField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### as_tensor
```python
LabelField.as_tensor(self, padding_lengths:Dict[str, int]) -> torch.Tensor
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
LabelField.empty_field(self)
```

So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
option ``TextFields``), we need a representation of an empty field of each type.  This
returns that.  This will only ever be called when we're to the point of calling
:func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
``count_vocab_items``, etc., being called on this empty field.

We make this an instance method instead of a static method so that if there is any state
in the Field, we can copy it over (e.g., the token indexers in ``TextField``).

