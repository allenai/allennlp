# allennlp.modules.attention.legacy_attention

## LegacyAttention
```python
LegacyAttention(self, similarity_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction=None, normalize:bool=True) -> None
```

Computes attention between a vector and a matrix using a similarity function.
This should be considered deprecated, as it consumes more memory than the specialized attention modules.

