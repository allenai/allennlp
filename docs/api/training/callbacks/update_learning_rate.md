# allennlp.training.callbacks.update_learning_rate

## UpdateLearningRate
```python
UpdateLearningRate(self, learning_rate_scheduler:allennlp.training.learning_rate_schedulers.learning_rate_scheduler.LearningRateScheduler) -> None
```

Callback that runs the learning rate scheduler.

Parameters
----------
learning_rate_scheduler : ``LearningRateScheduler``
    The scheduler to handler.

### get_training_state
```python
UpdateLearningRate.get_training_state(self) -> dict
```

We need to persist the learning_rate_scheduler state as training state.

