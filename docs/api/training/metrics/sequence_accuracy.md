# allennlp.training.metrics.sequence_accuracy

## SequenceAccuracy
```python
SequenceAccuracy(self) -> None
```

Sequence Top-K accuracy. Assumes integer labels, with
each item to be classified having a single correct class.

### get_metric
```python
SequenceAccuracy.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated accuracy.

### reset
```python
SequenceAccuracy.reset(self)
```

Reset any accumulators or internal state.

