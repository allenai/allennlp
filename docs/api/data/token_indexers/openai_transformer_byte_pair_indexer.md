# allennlp.data.token_indexers.openai_transformer_byte_pair_indexer

## text_standardize
```python
text_standardize(text)
```

Apply text standardization following original implementation.

## OpenaiTransformerBytePairIndexer
```python
OpenaiTransformerBytePairIndexer(self, encoder:Dict[str, int]=None, byte_pairs:List[Tuple[str, str]]=None, n_ctx:int=512, model_path:str=None, namespace:str='openai_transformer', tokens_to_add:List[str]=None, token_min_padding_length:int=0) -> None
```

Generates the indices for the byte-pair encoding used by
the OpenAI transformer language model: https://blog.openai.com/language-unsupervised/

This is unlike most of our TokenIndexers in that its
indexing is not based on a `Vocabulary` but on a fixed
set of mappings that are loaded by the constructor.

Note: recommend using ``OpenAIPreTokenizer`` tokenizer with this indexer,
as it applies the same text normalization as the original implementation.

Note 2: when ``tokens_to_add`` is not None, be sure to set
``n_special=len(tokens_to_add)`` in ``OpenaiTransformer``, otherwise
behavior is undefined.

### count_vocab_items
```python
OpenaiTransformerBytePairIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
OpenaiTransformerBytePairIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[int]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
OpenaiTransformerBytePairIndexer.get_padding_lengths(self, token:int) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### as_padded_tensor
```python
OpenaiTransformerBytePairIndexer.as_padded_tensor(self, tokens:Dict[str, List[int]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

