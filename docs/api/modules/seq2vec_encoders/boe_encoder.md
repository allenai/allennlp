# allennlp.modules.seq2vec_encoders.boe_encoder

## BagOfEmbeddingsEncoder
```python
BagOfEmbeddingsEncoder(self, embedding_dim:int, averaged:bool=False) -> None
```

A ``BagOfEmbeddingsEncoder`` is a simple :class:`Seq2VecEncoder` which simply sums the embeddings
of a sequence across the time dimension. The input to this module is of shape ``(batch_size, num_tokens,
embedding_dim)``, and the output is of shape ``(batch_size, embedding_dim)``.

Parameters
----------
embedding_dim : ``int``, required
    This is the input dimension to the encoder.
averaged : ``bool``, optional (default=``False``)
    If ``True``, this module will average the embeddings across time, rather than simply summing
    (ie. we will divide the summed embeddings by the length of the sentence).

### get_input_dim
```python
BagOfEmbeddingsEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
BagOfEmbeddingsEncoder.get_output_dim(self) -> int
```

Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
the shape of the returned tensor, but the last element of that shape.

