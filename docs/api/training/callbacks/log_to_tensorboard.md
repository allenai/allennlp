# allennlp.training.callbacks.log_to_tensorboard

## LogToTensorboard
```python
LogToTensorboard(self, tensorboard:allennlp.training.tensorboard_writer.TensorboardWriter, log_batch_size_period:int=None) -> None
```

Callback that handles all Tensorboard logging.

Parameters
----------
tensorboard : ``TensorboardWriter``
    The TensorboardWriter instance to write to.
log_batch_size_period : int, optional (default: None)
    If provided, we'll log the average batch sizes to Tensorboard
    every this-many batches.

