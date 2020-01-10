# allennlp.modules.matrix_attention.legacy_matrix_attention

## LegacyMatrixAttention
```python
LegacyMatrixAttention(self, similarity_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction=None) -> None
```

The legacy implementation of ``MatrixAttention``.

It should be considered deprecated as it uses much more memory than the newer specialized
``MatrixAttention`` modules.

Parameters
----------
similarity_function : ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
    The similarity function to use when computing the attention.

