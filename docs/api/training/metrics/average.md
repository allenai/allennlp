# allennlp.training.metrics.average

## Average
```python
Average(self) -> None
```

This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
computed in some fashion outside of a ``Metric``.  If you have some external code that computes
the metric for you, for instance, you can use this to report the average result using our
``Metric`` API.

### get_metric
```python
Average.get_metric(self, reset:bool=False)
```

Returns
-------
The average of all values that were passed to ``__call__``.

### reset
```python
Average.reset(self)
```

Reset any accumulators or internal state.

