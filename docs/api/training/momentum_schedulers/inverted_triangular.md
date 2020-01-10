# allennlp.training.momentum_schedulers.inverted_triangular

## InvertedTriangular
```python
InvertedTriangular(self, optimizer:torch.optim.optimizer.Optimizer, cool_down:int, warm_up:int, ratio:int=10, last_epoch:int=-1) -> None
```

Adjust momentum during training according to an inverted triangle-like schedule.

The momentum starts off high, then decreases linearly for ``cool_down`` epochs,
until reaching ``1 / ratio`` th of the original value. Then the momentum increases
linearly for ``warm_up`` epochs until reaching its original value again. If there
are still more epochs left over to train, the momentum will stay flat at the original
value.

