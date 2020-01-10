# allennlp.training.metrics.covariance

## Covariance
```python
Covariance(self) -> None
```

This ``Metric`` calculates the unbiased sample covariance between two tensors.
Each element in the two tensors is assumed to be a different observation of the
variable (i.e., the input tensors are implicitly flattened into vectors and the
covariance is calculated between the vectors).

This implementation is mostly modeled after the streaming_covariance function in Tensorflow. See:
https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3127

The following is copied from the Tensorflow documentation:

The algorithm used for this online computation is described in
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online .
Specifically, the formula used to combine two sample comoments is
`C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
The comoment for a single batch of data is simply `sum((x - E[x]) * (y - E[y]))`, optionally masked.

### get_metric
```python
Covariance.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated covariance.

### reset
```python
Covariance.reset(self)
```

Reset any accumulators or internal state.

