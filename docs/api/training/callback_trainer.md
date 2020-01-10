# allennlp.training.callback_trainer

The ``CallbackTrainer`` should be considered experimental code.
Its API may change at any time, and it may disappear altogether.

## CallbackTrainer
```python
CallbackTrainer(self, model:allennlp.models.model.Model, training_data:Iterable[allennlp.data.instance.Instance], iterator:allennlp.data.iterators.data_iterator.DataIterator, optimizer:torch.optim.optimizer.Optimizer, num_epochs:int=20, shuffle:bool=True, serialization_dir:Union[str, NoneType]=None, cuda_device:int=-1, callbacks:List[allennlp.training.callbacks.callback.Callback]=None, distributed:bool=False, rank:int=0, world_size:int=1) -> None
```

### generate_training_batches
```python
CallbackTrainer.generate_training_batches(self)
```

Generates one epoch worth of training data. Stores it in trainer instance variables
so that callbacks can access it.

### batch_loss
```python
CallbackTrainer.batch_loss(self, batch:Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], for_training:bool) -> torch.Tensor
```

Does a forward pass on the given batches and returns the ``loss`` value in the result.
If ``for_training`` is `True` also applies regularization penalty.

This is a method on the trainer so that it can be used both in training and validation
(which are handled separately).

### train_one_batch
```python
CallbackTrainer.train_one_batch(self, batch:Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> str
```

Handles the training for a single batch group.
Fires off the events BATCH_START, FORWARD, BACKWARD, and BATCH_END.

### train_one_epoch
```python
CallbackTrainer.train_one_epoch(self) -> None
```

Trains the model for a single epoch.
Fires off the events EPOCH_START and EPOCH_END,
and repeatedly calls self.train_one_batch().

### train
```python
CallbackTrainer.train(self) -> Dict[str, Any]
```

Trains the supplied model with the supplied parameters.
Fires off the events TRAINING_START and TRAINING END,
and repeatedly calls `self.train_one_epoch()`.

