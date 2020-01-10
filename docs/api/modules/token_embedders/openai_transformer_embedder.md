# allennlp.modules.token_embedders.openai_transformer_embedder

## OpenaiTransformerEmbedder
```python
OpenaiTransformerEmbedder(self, transformer:allennlp.modules.openai_transformer.OpenaiTransformer, top_layer_only:bool=False) -> None
```

Takes a byte-pair representation of a batch of sentences
(as produced by the ``OpenaiTransformerBytePairIndexer``)
and generates a `ScalarMix` of the corresponding contextual embeddings.



Parameters
----------
transformer : ``OpenaiTransformer``, required.
    The ``OpenaiTransformer`` module used for the embeddings.
top_layer_only : ``bool``, optional (default = ``False``)
    If ``True``, then only return the top layer instead of apply the scalar mix.

### get_output_dim
```python
OpenaiTransformerEmbedder.get_output_dim(self)
```

The last dimension of the output, not the shape.

### forward
```python
OpenaiTransformerEmbedder.forward(self, inputs:torch.Tensor, offsets:torch.Tensor=None) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``, required
    A ``(batch_size, num_timesteps)`` tensor representing the byte-pair encodings
    for the current batch.
offsets : ``torch.Tensor``, required
    A ``(batch_size, max_sequence_length)`` tensor representing the word offsets
    for the current batch.

Returns
-------
``[torch.Tensor]``
    An embedding representation of the input sequence
    having shape ``(batch_size, sequence_length, embedding_dim)``

