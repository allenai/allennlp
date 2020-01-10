# allennlp.modules.seq2seq_encoders.gated_cnn_encoder

## GatedCnnEncoder
```python
GatedCnnEncoder(self, input_dim:int, layers:Sequence[Sequence[Sequence[int]]], dropout:float=0.0, return_all_layers:bool=False) -> None
```

**This is work-in-progress and has not been fully tested yet. Use at your own risk!**

A ``Seq2SeqEncoder`` that uses a Gated CNN.

see

Language Modeling with Gated Convolutional Networks,  Yann N. Dauphin et al, ICML 2017
https://arxiv.org/abs/1612.08083

Convolutional Sequence to Sequence Learning, Jonas Gehring et al, ICML 2017
https://arxiv.org/abs/1705.03122

Some possibilities:

Each element of the list is wrapped in a residual block:
input_dim = 512
layers = [ [[4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]], [[4, 512], [4, 512]]
dropout = 0.05

A "bottleneck architecture"
input_dim = 512
layers = [ [[4, 512]], [[1, 128], [5, 128], [1, 512]], ... ]

An architecture with dilated convolutions
input_dim = 512
layers = [
[[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 16
[[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 31
[[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 46
[[2, 512, 1]], [[2, 512, 2]], [[2, 512, 4]], [[2, 512, 8]],   # receptive field == 57
]


Parameters
----------
input_dim : ``int``, required
    The dimension of the inputs.
layers : ``Sequence[Sequence[Sequence[int]]]``, required
    The layer dimensions for each ``ResidualBlock``.
dropout : ``float``, optional (default = 0.0)
    The dropout for each ``ResidualBlock``.
return_all_layers : ``bool``, optional (default = False)
    Whether to return all layers or just the last layer.

