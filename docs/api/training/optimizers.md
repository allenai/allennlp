# allennlp.training.optimizers

AllenNLP just uses
`PyTorch optimizers <https://pytorch.org/docs/master/optim.html>`_ ,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available optimizers are

* `"adadelta" <https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta>`_
* `"adagrad" <https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad>`_
* `"adam" <https://pytorch.org/docs/master/optim.html#torch.optim.Adam>`_
* `"adamw" <https://pytorch.org/docs/master/optim.html#torch.optim.AdamW>`_
* `"huggingface_adamw"
  <https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW>`_
* `"sparse_adam" <https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam>`_
* `"sgd" <https://pytorch.org/docs/master/optim.html#torch.optim.SGD>`_
* `"rmsprop <https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop>`_
* `"adamax <https://pytorch.org/docs/master/optim.html#torch.optim.Adamax>`_
* `"averaged_sgd <https://pytorch.org/docs/master/optim.html#torch.optim.ASGD>`_

## Optimizer
```python
Optimizer(self, /, *args, **kwargs)
```

This class just allows us to implement ``Registrable`` for Pytorch Optimizers.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
## DenseSparseAdam
```python
DenseSparseAdam(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```

NOTE: This class has been copied verbatim from the separate Dense and
Sparse versions of Adam in Pytorch.

Implements Adam algorithm with dense & sparse gradients.
It has been proposed in Adam: A Method for Stochastic Optimization.

Parameters
----------
params : ``iterable``
    iterable of parameters to optimize or dicts defining parameter groups
lr : ``float``, optional (default: 1e-3)
    The learning rate.
betas : ``Tuple[float, float]``, optional (default: (0.9, 0.999))
    coefficients used for computing running averages of gradient
    and its square.
eps : ``float``, optional, (default: 1e-8)
    A term added to the denominator to improve numerical stability.

### step
```python
DenseSparseAdam.step(self, closure=None)
```

Performs a single optimization step.

Parameters
----------
closure : ``callable``, optional.
    A closure that reevaluates the model and returns the loss.

