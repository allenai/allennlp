# allennlp.modules.token_embedders.pretrained_transformer_embedder

## PretrainedTransformerEmbedder
```python
PretrainedTransformerEmbedder(self, model_name:str) -> None
```

Uses a pretrained model from ``transformers`` as a ``TokenEmbedder``.

### get_output_dim
```python
PretrainedTransformerEmbedder.get_output_dim(self)
```

Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
token.  This is `not` the shape of the returned tensor, but the last element of that shape.

