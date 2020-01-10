# allennlp.data.token_indexers.pretrained_transformer_indexer

## PretrainedTransformerIndexer
```python
PretrainedTransformerIndexer(self, model_name:str, namespace:str='tags', token_min_padding_length:int=0) -> None
```

This ``TokenIndexer`` assumes that Tokens already have their indexes in them (see ``text_id`` field).
We still require ``model_name`` because we want to form allennlp vocabulary from pretrained one.
This ``Indexer`` is only really appropriate to use if you've also used a
corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

Parameters
----------
model_name : ``str``
    The name of the ``transformers`` model to use.
namespace : ``str``, optional (default=``tags``)
    We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
    We use a somewhat confusing default value of ``tags`` so that we do not add padding or UNK
    tokens to this namespace, which would break on loading because we wouldn't find our default
    OOV token.

### count_vocab_items
```python
PretrainedTransformerIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
PretrainedTransformerIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[int]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
PretrainedTransformerIndexer.get_padding_lengths(self, token:int) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### as_padded_tensor
```python
PretrainedTransformerIndexer.as_padded_tensor(self, tokens:Dict[str, List[int]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

