# allennlp.training.moving_average

## MovingAverage
```python
MovingAverage(self, parameters:Iterable[Tuple[str, torch.Tensor]]) -> None
```

Tracks a moving average of model parameters.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### apply
```python
MovingAverage.apply(self, num_updates:Union[int, NoneType]=None)
```

Update the moving averages based on the latest values of the parameters.

### assign_average_value
```python
MovingAverage.assign_average_value(self) -> None
```

Replace all the parameter values with the averages.
Save the current parameter values to restore later.

### restore
```python
MovingAverage.restore(self) -> None
```

Restore the backed-up (non-average) parameter values.

## ExponentialMovingAverage
```python
ExponentialMovingAverage(self, parameters:Iterable[Tuple[str, torch.Tensor]], decay:float=0.9999, numerator:float=1.0, denominator:float=10.0) -> None
```

Create shadow variables and maintain exponential moving average for model parameters.

Parameters
----------
parameters : ``Iterable[Tuple[str, Parameter]]``, required
    The parameters whose averages we'll be tracking.
decay : ``float``, optional (default = 0.9999)
    The decay rate that will be used if `num_updates` is not passed
    (and that will be used as an upper bound if `num_updates` is passed).
numerator : ``float``, optional (default = 1.0)
    The numerator used to compute the decay rate if ``num_updates`` is passed.
denominator : ``float``, optional (default = 10.0)
    The denominator used to compute the decay rate if ``num_updates`` is passed.

### apply
```python
ExponentialMovingAverage.apply(self, num_updates:Union[int, NoneType]=None) -> None
```

Apply exponential moving average to `named_parameters` if specified,
or we will apply this to all the trainable parameters of the model.

The optional `num_updates` parameter allows one to tweak the decay rate
dynamically. If passed, the actual decay rate used is:

    `min(decay, (numerator + num_updates) / (denominator + num_updates))`

(This logic is based on the Tensorflow exponential moving average
 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)

