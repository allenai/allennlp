# allennlp.training.metrics.mean_absolute_error

## MeanAbsoluteError
```python
MeanAbsoluteError(self) -> None
```

This ``Metric`` calculates the mean absolute error (MAE) between two tensors.

### get_metric
```python
MeanAbsoluteError.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated mean absolute error.

### reset
```python
MeanAbsoluteError.reset(self)
```

Reset any accumulators or internal state.

