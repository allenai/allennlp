# allennlp.training.learning_rate_schedulers.cosine

## CosineWithRestarts
```python
CosineWithRestarts(self, optimizer:torch.optim.optimizer.Optimizer, t_initial:int, t_mul:float=1.0, eta_min:float=0.0, eta_mul:float=1.0, last_epoch:int=-1) -> None
```

Cosine annealing with restarts.

This is described in the paper https://arxiv.org/abs/1608.03983. Note that early
stopping should typically be avoided when using this schedule.

Parameters
----------
optimizer : ``torch.optim.Optimizer``
t_initial : ``int``
    The number of iterations (epochs) within the first cycle.
t_mul : ``float``, optional (default=1)
    Determines the number of iterations (epochs) in the i-th decay cycle,
    which is the length of the last cycle multiplied by ``t_mul``.
eta_min : ``float``, optional (default=0)
    The minimum learning rate.
eta_mul : ``float``, optional (default=1)
    Determines the initial learning rate for the i-th decay cycle, which is the
    last initial learning rate multiplied by ``m_mul``.
last_epoch : ``int``, optional (default=-1)
    The index of the last epoch. This is used when restarting.

### get_values
```python
CosineWithRestarts.get_values(self)
```
Get updated learning rate.
