# allennlp.modules.residual_with_layer_dropout

## ResidualWithLayerDropout
```python
ResidualWithLayerDropout(self, undecayed_dropout_prob:float=0.5) -> None
```

A residual connection with the layer dropout technique `Deep Networks with Stochastic
Depth <https://arxiv.org/pdf/1603.09382.pdf>`_ .

This module accepts the input and output of a layer, decides whether this layer should
be stochastically dropped, returns either the input or output + input. During testing,
it will re-calibrate the outputs of this layer by the expected number of times it
participates in training.

### forward
```python
ResidualWithLayerDropout.forward(self, layer_input:torch.Tensor, layer_output:torch.Tensor, layer_index:int=None, total_layers:int=None) -> torch.Tensor
```

Apply dropout to this layer, for this whole mini-batch.
dropout_prob = layer_index / total_layers * undecayed_dropout_prob if layer_idx and
total_layers is specified, else it will use the undecayed_dropout_prob directly.

Parameters
----------
layer_input ``torch.FloatTensor`` required
    The input tensor of this layer.
layer_output ``torch.FloatTensor`` required
    The output tensor of this layer, with the same shape as the layer_input.
layer_index ``int``
    The layer index, starting from 1. This is used to calcuate the dropout prob
    together with the `total_layers` parameter.
total_layers ``int``
    The total number of layers.

Returns
-------
output : ``torch.FloatTensor``
    A tensor with the same shape as `layer_input` and `layer_output`.

