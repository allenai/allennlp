# allennlp.data.token_indexers.dep_label_indexer

## DepLabelIndexer
```python
DepLabelIndexer(self, namespace:str='dep_labels', token_min_padding_length:int=0) -> None
```

This :class:`TokenIndexer` represents tokens by their syntactic dependency label, as determined
by the ``dep_`` field on ``Token``.

Parameters
----------
namespace : ``str``, optional (default=``dep_labels``)
    We will use this namespace in the :class:`Vocabulary` to map strings to indices.
token_min_padding_length : ``int``, optional (default=``0``)
    See :class:`TokenIndexer`.

### count_vocab_items
```python
DepLabelIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
DepLabelIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[int]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
DepLabelIndexer.get_padding_lengths(self, token:int) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### as_padded_tensor
```python
DepLabelIndexer.as_padded_tensor(self, tokens:Dict[str, List[int]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

