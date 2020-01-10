# allennlp.training.trainer

## Trainer
```python
Trainer(self, model:allennlp.models.model.Model, optimizer:torch.optim.optimizer.Optimizer, iterator:allennlp.data.iterators.data_iterator.DataIterator, train_dataset:Iterable[allennlp.data.instance.Instance], validation_dataset:Union[Iterable[allennlp.data.instance.Instance], NoneType]=None, patience:Union[int, NoneType]=None, validation_metric:str='-loss', validation_iterator:allennlp.data.iterators.data_iterator.DataIterator=None, shuffle:bool=True, num_epochs:int=20, serialization_dir:Union[str, NoneType]=None, num_serialized_models_to_keep:int=20, keep_serialized_model_every_num_seconds:int=None, checkpointer:allennlp.training.checkpointer.Checkpointer=None, model_save_interval:float=None, cuda_device:int=-1, grad_norm:Union[float, NoneType]=None, grad_clipping:Union[float, NoneType]=None, learning_rate_scheduler:Union[allennlp.training.learning_rate_schedulers.learning_rate_scheduler.LearningRateScheduler, NoneType]=None, momentum_scheduler:Union[allennlp.training.momentum_schedulers.momentum_scheduler.MomentumScheduler, NoneType]=None, summary_interval:int=100, histogram_interval:int=None, should_log_parameter_statistics:bool=True, should_log_learning_rate:bool=False, log_batch_size_period:Union[int, NoneType]=None, moving_average:Union[allennlp.training.moving_average.MovingAverage, NoneType]=None, distributed:bool=False, rank:int=0, world_size:int=1, num_gradient_accumulation_steps:int=1) -> None
```

### batch_loss
```python
Trainer.batch_loss(self, batch:Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], for_training:bool) -> torch.Tensor
```

Does a forward pass on the given batches and returns the ``loss`` value in the result.
If ``for_training`` is `True` also applies regularization penalty.

### train
```python
Trainer.train(self) -> Dict[str, Any]
```

Trains the supplied model with the supplied parameters.

