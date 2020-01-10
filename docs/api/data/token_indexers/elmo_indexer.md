# allennlp.data.token_indexers.elmo_indexer

## ELMoCharacterMapper
```python
ELMoCharacterMapper(self, tokens_to_add:Dict[str, int]=None) -> None
```

Maps individual tokens to sequences of character ids, compatible with ELMo.
To be consistent with previously trained models, we include it here as special of existing
character indexers.

We allow to add optional additional special tokens with designated
character ids with ``tokens_to_add``.

### beginning_of_sentence_character
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### beginning_of_sentence_characters
list() -> new empty list
list(iterable) -> new list initialized from iterable's items
### beginning_of_word_character
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### bos_token
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### end_of_sentence_character
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### end_of_sentence_characters
list() -> new empty list
list(iterable) -> new list initialized from iterable's items
### end_of_word_character
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### eos_token
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### max_word_length
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
### padding_character
int(x=0) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
## ELMoTokenCharactersIndexer
```python
ELMoTokenCharactersIndexer(self, namespace:str='elmo_characters', tokens_to_add:Dict[str, int]=None, token_min_padding_length:int=0) -> None
```

Convert a token to an array of character ids to compute ELMo representations.

Parameters
----------
namespace : ``str``, optional (default=``elmo_characters``)
tokens_to_add : ``Dict[str, int]``, optional (default=``None``)
    If not None, then provides a mapping of special tokens to character
    ids. When using pre-trained models, then the character id must be
    less then 261, and we recommend using un-used ids (e.g. 1-32).
token_min_padding_length : ``int``, optional (default=``0``)
    See :class:`TokenIndexer`.

### count_vocab_items
```python
ELMoTokenCharactersIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
ELMoTokenCharactersIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[List[int]]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
ELMoTokenCharactersIndexer.get_padding_lengths(self, token:List[int]) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### as_padded_tensor
```python
ELMoTokenCharactersIndexer.as_padded_tensor(self, tokens:Dict[str, List[List[int]]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

