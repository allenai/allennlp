# allennlp.modules.drop_connect

## DropConnect
```python
DropConnect(self, module:torch.nn.modules.module.Module, parameter_regex:str, dropout:float=0.0) -> None
```

DropConnect module described in: `"Regularization of Neural Networks using DropConnect"
<https://www.semanticscholar.org/paper/Regularization-of-Neural-Networks-using-DropConnect-Wan-Zeiler/38f35dd624cd1cf827416e31ac5e0e0454028eca>`_
by Wan et al., 2013. Applies dropout to module parameters instead of module outputs.

Parameters
==========
module : ``torch.nn.Module``, required
    Module to apply weight dropout to.
parameter_regex : ``str``, required
    Regular expression identifying which parameters to apply weight dropout to.
dropout : ``float``, optional (default = 0.0)
    Probability that a given weight is dropped.

