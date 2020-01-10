# allennlp.training.learning_rate_schedulers.noam

## NoamLR
```python
NoamLR(self, optimizer:torch.optim.optimizer.Optimizer, model_size:int, warmup_steps:int, factor:float=1.0, last_epoch:int=-1) -> None
```

Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
to the inverse square root of the step number, scaled by the inverse square root of the
dimensionality of the model. Time will tell if this is just madness or it's actually important.

Parameters
----------
model_size : ``int``, required.
    The hidden size parameter which dominates the number of parameters in your model.
warmup_steps : ``int``, required.
    The number of steps to linearly increase the learning rate.
factor : ``float``, optional (default = 1.0).
    The overall scale factor for the learning rate decay.

