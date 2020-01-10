# allennlp.training.tensorboard_writer

## TensorboardWriter
```python
TensorboardWriter(self, get_batch_num_total:Callable[[], int], serialization_dir:Union[str, NoneType]=None, summary_interval:int=100, histogram_interval:int=None, should_log_parameter_statistics:bool=True, should_log_learning_rate:bool=False) -> None
```

Class that handles Tensorboard (and other) logging.

Parameters
----------
get_batch_num_total : Callable[[], int]
    A thunk that returns the number of batches so far. Most likely this will
    be a closure around an instance variable in your ``Trainer`` class.
serialization_dir : str, optional (default = None)
    If provided, this is where the Tensorboard logs will be written.
summary_interval : int, optional (default = 100)
    Most statistics will be written out only every this many batches.
histogram_interval : int, optional (default = None)
    If provided, activation histograms will be written out every this many batches.
    If None, activation histograms will not be written out.
should_log_parameter_statistics : bool, optional (default = True)
    Whether to log parameter statistics.
should_log_learning_rate : bool, optional (default = False)
    Whether to log learning rate.

### log_parameter_and_gradient_statistics
```python
TensorboardWriter.log_parameter_and_gradient_statistics(self, model:allennlp.models.model.Model, batch_grad_norm:float) -> None
```

Send the mean and std of all parameters and gradients to tensorboard, as well
as logging the average gradient norm.

### log_learning_rates
```python
TensorboardWriter.log_learning_rates(self, model:allennlp.models.model.Model, optimizer:torch.optim.optimizer.Optimizer)
```

Send current parameter specific learning rates to tensorboard

### log_histograms
```python
TensorboardWriter.log_histograms(self, model:allennlp.models.model.Model, histogram_parameters:Set[str]) -> None
```

Send histograms of parameters to tensorboard.

### log_metrics
```python
TensorboardWriter.log_metrics(self, train_metrics:dict, val_metrics:dict=None, epoch:int=None, log_to_console:bool=False) -> None
```

Sends all of the train metrics (and validation metrics, if provided) to tensorboard.

### close
```python
TensorboardWriter.close(self) -> None
```

Calls the ``close`` method of the ``SummaryWriter`` s which makes sure that pending
scalars are flushed to disk and the tensorboard event files are closed properly.

