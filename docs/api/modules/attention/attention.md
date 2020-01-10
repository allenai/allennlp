# allennlp.modules.attention.attention

An *attention* module that computes the similarity between
an input vector and the rows of a matrix.

## Attention
```python
Attention(self, normalize:bool=True) -> None
```

An ``Attention`` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
and then (optionally) perform a softmax over rows using those computed similarities.


Inputs:

- vector: shape ``(batch_size, embedding_dim)``
- matrix: shape ``(batch_size, num_rows, embedding_dim)``
- matrix_mask: shape ``(batch_size, num_rows)``, specifying which rows are just padding.

Output:

- attention: shape ``(batch_size, num_rows)``.

Parameters
----------
normalize : ``bool``, optional (default : ``True``)
    If true, we normalize the computed similarities with a softmax, to return a probability
    distribution for your attention.  If false, this is just computing a similarity score.

### forward
```python
Attention.forward(self, vector:torch.Tensor, matrix:torch.Tensor, matrix_mask:torch.Tensor=None) -> torch.Tensor
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

