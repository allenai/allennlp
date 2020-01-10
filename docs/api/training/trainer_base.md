# allennlp.training.trainer_base

A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.

## TrainerBase
```python
TrainerBase(self, serialization_dir:str, cuda_device:int=-1, distributed:bool=False, rank:int=0, world_size:int=1) -> None
```

The base class for an AllenNLP trainer. It can do pretty much
anything you want. Your subclass should implement ``train``
and also probably ``from_params``.

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
### train
```python
TrainerBase.train(self) -> Dict[str, Any]
```

Train a model and return the results.

