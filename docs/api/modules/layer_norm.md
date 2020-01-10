# allennlp.modules.layer_norm

## LayerNorm
```python
LayerNorm(self, dimension:int, eps:float=1e-06) -> None
```

An implementation of `Layer Normalization
<https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

Layer Normalization stabilises the training of deep neural networks by
normalising the outputs of neurons from a particular layer. It computes:

output = (gamma * (tensor - mean) / (std + eps)) + beta

Parameters
----------
dimension : ``int``, required.
    The dimension of the layer output to normalize.
eps : ``float``, optional, (default = 1e-6)
    An epsilon to prevent dividing by zero in the case
    the layer has zero variance.

Returns
-------
The normalized layer output.

