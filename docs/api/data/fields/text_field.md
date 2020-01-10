# allennlp.data.fields.text_field

A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.

## TextField
```python
TextField(self, tokens:List[allennlp.data.tokenizers.token.Token], token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]) -> None
```

This ``Field`` represents a list of string tokens.  Before constructing this object, you need
to tokenize raw strings using a :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer`.

Because string tokens can be represented as indexed arrays in a number of ways, we also take a
dictionary of :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer`
objects that will be used to convert the tokens into indices.
Each ``TokenIndexer`` could represent each token as a single ID, or a list of character IDs, or
something else.

This field will get converted into a dictionary of arrays, one for each ``TokenIndexer``.  A
``SingleIdTokenIndexer`` produces an array of shape (num_tokens,), while a
``TokenCharactersIndexer`` produces an array of shape (num_tokens, num_characters).

### count_vocab_items
```python
TextField.count_vocab_items(self, counter:Dict[str, Dict[str, int]])
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
TextField.index(self, vocab:allennlp.data.vocabulary.Vocabulary)
```

Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
This `modifies` the ``Field`` object, it does not return anything.

If your ``Field`` does not have any strings that need to be converted into indices, you do
not need to implement this method.

### get_padding_lengths
```python
TextField.get_padding_lengths(self) -> Dict[str, int]
```

The ``TextField`` has a list of ``Tokens``, and each ``Token`` gets converted into arrays by
(potentially) several ``TokenIndexers``.  This method gets the max length (over tokens)
associated with each of these arrays.

### sequence_length
```python
TextField.sequence_length(self) -> int
```

How many elements are there in this sequence?

### as_tensor
```python
TextField.as_tensor(self, padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
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
TextField.batch_tensors(self, tensor_list:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]
```

Takes the output of ``Field.as_tensor()`` from a list of ``Instances`` and merges it into
one batched tensor for this ``Field``.  The default implementation here in the base class
handles cases where ``as_tensor`` returns a single torch tensor per instance.  If your
subclass returns something other than this, you need to override this method.

This operation does not modify ``self``, but in some cases we need the information
contained in ``self`` in order to perform the batching, so this is an instance method, not
a class method.

