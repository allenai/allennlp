# allennlp.training.metrics.categorical_accuracy

## CategoricalAccuracy
```python
CategoricalAccuracy(self, top_k:int=1, tie_break:bool=False) -> None
```

Categorical Top-K accuracy. Assumes integer labels, with
each item to be classified having a single correct class.
Tie break enables equal distribution of scores among the
classes with same maximum predicted scores.

### get_metric
```python
CategoricalAccuracy.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated accuracy.

### reset
```python
CategoricalAccuracy.reset(self)
```

Reset any accumulators or internal state.

