# allennlp.data.token_indexers.token_characters_indexer

## TokenCharactersIndexer
```python
TokenCharactersIndexer(self, namespace:str='token_characters', character_tokenizer:allennlp.data.tokenizers.character_tokenizer.CharacterTokenizer=<allennlp.data.tokenizers.character_tokenizer.CharacterTokenizer object at 0x131c63e48>, start_tokens:List[str]=None, end_tokens:List[str]=None, min_padding_length:int=0, token_min_padding_length:int=0) -> None
```

This :class:`TokenIndexer` represents tokens as lists of character indices.

Parameters
----------
namespace : ``str``, optional (default=``token_characters``)
    We will use this namespace in the :class:`Vocabulary` to map the characters in each token
    to indices.
character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
    We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
    options for byte encoding and other things.  The default here is to instantiate a
    ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
    retains casing.
start_tokens : ``List[str]``, optional (default=``None``)
    These are prepended to the tokens provided to ``tokens_to_indices``.
end_tokens : ``List[str]``, optional (default=``None``)
    These are appended to the tokens provided to ``tokens_to_indices``.
min_padding_length : ``int``, optional (default=``0``)
    We use this value as the minimum length of padding. Usually used with :class:``CnnEncoder``, its
    value should be set to the maximum value of ``ngram_filter_sizes`` correspondingly.
token_min_padding_length : ``int``, optional (default=``0``)
    See :class:`TokenIndexer`.

### count_vocab_items
```python
TokenCharactersIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
TokenCharactersIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[List[int]]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
TokenCharactersIndexer.get_padding_lengths(self, token:List[int]) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### get_padding_token
```python
TokenCharactersIndexer.get_padding_token(self) -> List[int]
```

Deprecated. Please just implement the padding token in `as_padded_tensor` instead.
TODO(Mark): remove in 1.0 release. This is only a concrete implementation to preserve
backward compatability, otherwise it would be abstract.

When we need to add padding tokens, what should they look like?  This method returns a
"blank" token of whatever type is returned by :func:`tokens_to_indices`.

### as_padded_tensor
```python
TokenCharactersIndexer.as_padded_tensor(self, tokens:Dict[str, List[List[int]]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

