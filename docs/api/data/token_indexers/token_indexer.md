# allennlp.data.token_indexers.token_indexer

## TokenIndexer
```python
TokenIndexer(self, token_min_padding_length:int=0) -> None
```

A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.
This class both converts strings into numerical values, with the help of a
:class:`~allennlp.data.vocabulary.Vocabulary`, and it produces actual arrays.

Tokens can be represented as single IDs (e.g., the word "cat" gets represented by the number
34), or as lists of character IDs (e.g., "cat" gets represented by the numbers [23, 10, 18]),
or in some other way that you can come up with (e.g., if you have some structured input you
want to represent in a special way in your data arrays, you can do that here).

Parameters
----------
token_min_padding_length : ``int``, optional (default=``0``)
    The minimum padding length required for the :class:`TokenIndexer`. For example,
    the minimum padding length of :class:`SingleIdTokenIndexer` is the largest size of
    filter when using :class:`CnnEncoder`.
    Note that if you set this for one TokenIndexer, you likely have to set it for all
    :class:`TokenIndexer` for the same field, otherwise you'll get mismatched tensor sizes.

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
### has_warned_for_as_padded_tensor
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
### count_vocab_items
```python
TokenIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
TokenIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[~TokenType]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_token
```python
TokenIndexer.get_padding_token(self) -> ~TokenType
```

Deprecated. Please just implement the padding token in `as_padded_tensor` instead.
TODO(Mark): remove in 1.0 release. This is only a concrete implementation to preserve
backward compatability, otherwise it would be abstract.

When we need to add padding tokens, what should they look like?  This method returns a
"blank" token of whatever type is returned by :func:`tokens_to_indices`.

### get_padding_lengths
```python
TokenIndexer.get_padding_lengths(self, token:~TokenType) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### get_token_min_padding_length
```python
TokenIndexer.get_token_min_padding_length(self) -> int
```

This method returns the minimum padding length required for this TokenIndexer.
For example, the minimum padding length of `SingleIdTokenIndexer` is the largest
size of filter when using `CnnEncoder`.

### as_padded_tensor
```python
TokenIndexer.as_padded_tensor(self, tokens:Dict[str, List[~TokenType]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

### pad_token_sequence
```python
TokenIndexer.pad_token_sequence(self, tokens:Dict[str, List[~TokenType]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, ~TokenType]
```

Deprecated. Please use `as_padded_tensor` instead.
TODO(Mark): remove in 1.0 release.

### get_keys
```python
TokenIndexer.get_keys(self, index_name:str) -> List[str]
```

Return a list of the keys this indexer return from ``tokens_to_indices``.

