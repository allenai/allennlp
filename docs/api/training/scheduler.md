# allennlp.training.scheduler

## Scheduler
```python
Scheduler(self, optimizer:torch.optim.optimizer.Optimizer, param_group_field:str, last_epoch:int=-1) -> None
```

A ``Scheduler`` is a generalization of PyTorch learning rate schedulers.

A scheduler can be used to update any field in an optimizer's parameter groups,
not just the learning rate.

During training using the AllenNLP `Trainer`, this is the API and calling
sequence for ``step`` and ``step_batch``::

   scheduler = ... # creates scheduler, calls self.step(epoch=-1) in __init__

   batch_num_total = 0
   for epoch in range(num_epochs):
       for batch in batchs_in_epoch:
           # compute loss, update parameters with current learning rates
           # call step_batch AFTER updating parameters
           batch_num_total += 1
           scheduler.step_batch(batch_num_total)
       # call step() at the END of each epoch
       scheduler.step(validation_metrics, epoch)

### state_dict
```python
Scheduler.state_dict(self) -> Dict[str, Any]
```

Returns the state of the scheduler as a ``dict``.

### load_state_dict
```python
Scheduler.load_state_dict(self, state_dict:Dict[str, Any]) -> None
```

Load the schedulers state.

Parameters
----------
state_dict : ``Dict[str, Any]``
    Scheduler state. Should be an object returned from a call to ``state_dict``.

### step_batch
```python
Scheduler.step_batch(self, batch_num_total:int=None) -> None
```

By default, a scheduler is assumed to only update every epoch, not every batch.
So this does nothing unless it's overriden.

