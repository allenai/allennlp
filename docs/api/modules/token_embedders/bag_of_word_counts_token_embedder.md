# allennlp.modules.token_embedders.bag_of_word_counts_token_embedder

## BagOfWordCountsTokenEmbedder
```python
BagOfWordCountsTokenEmbedder(self, vocab:allennlp.data.vocabulary.Vocabulary, vocab_namespace:str, projection_dim:int=None, ignore_oov:bool=False) -> None
```

Represents a sequence of tokens as a bag of (discrete) word ids, as it was done
in the pre-neural days.

Each sequence gets a vector of length vocabulary size, where the i'th entry in the vector
corresponds to number of times the i'th token in the vocabulary appears in the sequence.

By default, we ignore padding tokens.

Parameters
----------
vocab : ``Vocabulary``
vocab_namespace : ``str``
    namespace of vocabulary to embed
projection_dim : ``int``, optional (default = ``None``)
    if specified, will project the resulting bag of words representation
    to specified dimension.
ignore_oov : ``bool``, optional (default = ``False``)
    If true, we ignore the OOV token.

### forward
```python
BagOfWordCountsTokenEmbedder.forward(self, inputs:torch.Tensor) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``
    Shape ``(batch_size, timesteps, sequence_length)`` of word ids
    representing the current batch.

Returns
-------
The bag-of-words representations for the input sequence, shape
``(batch_size, vocab_size)``

### from_params
```python
BagOfWordCountsTokenEmbedder.from_params(vocab:allennlp.data.vocabulary.Vocabulary, params:allennlp.common.params.Params) -> 'BagOfWordCountsTokenEmbedder'
```

we look for a ``vocab_namespace`` key in the parameter dictionary
to know which vocabulary to use.

