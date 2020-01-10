# allennlp.training.metrics.spearman_correlation

## SpearmanCorrelation
```python
SpearmanCorrelation(self) -> None
```

This ``Metric`` calculates the sample Spearman correlation coefficient (r)
between two tensors. Each element in the two tensors is assumed to be
a different observation of the variable (i.e., the input tensors are
implicitly flattened into vectors and the correlation is calculated
between the vectors).

https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

### get_metric
```python
SpearmanCorrelation.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated sample Spearman correlation.

### reset
```python
SpearmanCorrelation.reset(self)
```

Reset any accumulators or internal state.

