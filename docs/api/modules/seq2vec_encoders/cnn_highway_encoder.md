# allennlp.modules.seq2vec_encoders.cnn_highway_encoder

## CnnHighwayEncoder
```python
CnnHighwayEncoder(self, embedding_dim:int, filters:Sequence[Sequence[int]], num_highway:int, projection_dim:int, activation:str='relu', projection_location:str='after_highway', do_layer_norm:bool=False) -> None
```

The character CNN + highway encoder from Kim et al "Character aware neural language models"
https://arxiv.org/abs/1508.06615
with an optional projection.

Parameters
----------
embedding_dim : ``int``, required
    The dimension of the initial character embedding.
filters : ``Sequence[Sequence[int]]``, required
    A sequence of pairs (filter_width, num_filters).
num_highway : ``int``, required
    The number of highway layers.
projection_dim : ``int``, required
    The output dimension of the projection layer.
activation : ``str``, optional (default = 'relu')
    The activation function for the convolutional layers.
projection_location : ``str``, optional (default = 'after_highway')
    Where to apply the projection layer. Valid values are
    'after_highway', 'after_cnn', and None.

### forward
```python
CnnHighwayEncoder.forward(self, inputs:torch.Tensor, mask:torch.Tensor) -> Dict[str, torch.Tensor]
```

Compute context insensitive token embeddings for ELMo representations.

Parameters
----------
inputs:
    Shape ``(batch_size, num_characters, embedding_dim)``
    Character embeddings representing the current batch.
mask:
    Shape ``(batch_size, num_characters)``
    Currently unused. The mask for characters is implicit. See TokenCharactersEncoder.forward.

Returns
-------
``encoding``:
    Shape ``(batch_size, projection_dim)`` tensor with context-insensitive token representations.

