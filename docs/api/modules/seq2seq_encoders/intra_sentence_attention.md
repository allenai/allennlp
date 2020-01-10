# allennlp.modules.seq2seq_encoders.intra_sentence_attention

## IntraSentenceAttentionEncoder
```python
IntraSentenceAttentionEncoder(self, input_dim:int, projection_dim:int=None, similarity_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction=DotProductSimilarity(), num_attention_heads:int=1, combination:str='1,2', output_dim:int=None) -> None
```

An ``IntraSentenceAttentionEncoder`` is a :class:`Seq2SeqEncoder` that merges the original word
representations with an attention (for each word) over other words in the sentence.  As a
:class:`Seq2SeqEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
input_dim)``, and the output is of shape ``(batch_size, num_tokens, output_dim)``.

We compute the attention using a configurable :class:`SimilarityFunction`, which could have
multiple attention heads.  The operation for merging the original representations with the
attended representations is also configurable (e.g., you can concatenate them, add them,
multiply them, etc.).

Parameters
----------
input_dim : ``int`` required
    The dimension of the vector for each element in the input sequence;
    ``input_tensor.size(-1)``.
projection_dim : ``int``, optional
    If given, we will do a linear projection of the input sequence to this dimension before
    performing the attention-weighted sum.
similarity_function : ``SimilarityFunction``, optional
    The similarity function to use when computing attentions.  Default is to use a dot product.
num_attention_heads : ``int``, optional
    If this is greater than one (default is 1), we will split the input into several "heads" to
    compute multi-headed weighted sums.  Must be used with a multi-headed similarity function,
    and you almost certainly want to do a projection in conjunction with the multiple heads.
combination : ``str``, optional
    This string defines how we merge the original word representations with the result of the
    intra-sentence attention.  This will be passed to
    :func:`~allennlp.nn.util.combine_tensors`; see that function for more detail on exactly how
    this works, but some simple examples are ``"1,2"`` for concatenation (the default),
    ``"1+2"`` for adding the two, or ``"2"`` for only keeping the attention representation.
output_dim : ``int``, optional (default = None)
    The dimension of an optional output projection.

### get_input_dim
```python
IntraSentenceAttentionEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2SeqEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
IntraSentenceAttentionEncoder.get_output_dim(self) -> int
```

Returns the dimension of each vector in the sequence output by this ``Seq2SeqEncoder``.
This is `not` the shape of the returned tensor, but the last element of that shape.

### is_bidirectional
```python
IntraSentenceAttentionEncoder.is_bidirectional(self)
```

Returns ``True`` if this encoder is bidirectional.  If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.

### forward
```python
IntraSentenceAttentionEncoder.forward(self, tokens:torch.Tensor, mask:torch.Tensor)
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

