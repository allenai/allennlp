# allennlp.training.learning_rate_schedulers.slanted_triangular

## SlantedTriangular
```python
SlantedTriangular(self, optimizer:torch.optim.optimizer.Optimizer, num_epochs:int, num_steps_per_epoch:int, cut_frac:float=0.1, ratio:int=32, last_epoch:int=-1, gradual_unfreezing:bool=False, discriminative_fine_tuning:bool=False, decay_factor:float=0.38) -> None
```

Implements the Slanted Triangular Learning Rate schedule with optional gradual
unfreezing. The schedule corresponds to first linearly increasing the learning
rate and annealing the learning based on a fixed ratio.

If we gradually unfreeze, then in the first epoch of training, only the top
layer is trained; in the second epoch, the top two layers are trained, etc.
During freezing, the learning rate is increased and annealed over one epoch.
After freezing finished, the learning rate is increased and annealed over
the remaining training iterations.

Note that with this schedule, early stopping should typically be avoided.

Parameters
----------
num_epochs : ``int``, required.
    The total number of epochs for which the model should be trained.
num_steps_per_epoch : ``int``, required.
    The number of steps (updates, batches) per training epoch.
cut_frac : ``float``, optional (default = 0.1).
    The fraction of the steps to increase the learning rate.
ratio : ``float``, optional (default = 32).
    The ratio of the smallest to the (largest) base learning rate.
gradual_unfreezing : ``bool``, optional (default = False).
    Whether gradual unfreezing should be used.
discriminative_fine_tuning : ``bool``, optional (default = False).
    Whether discriminative fine-tuning (different learning rates per layer)
    are used.
decay_factor : ``float``, optional (default = 0.38).
    The decay factor by which the learning rate is reduced with
    discriminative fine-tuning when going a layer deeper.

