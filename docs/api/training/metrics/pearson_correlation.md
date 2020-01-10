# allennlp.training.metrics.pearson_correlation

## PearsonCorrelation
```python
PearsonCorrelation(self) -> None
```

This ``Metric`` calculates the sample Pearson correlation coefficient (r)
between two tensors. Each element in the two tensors is assumed to be
a different observation of the variable (i.e., the input tensors are
implicitly flattened into vectors and the correlation is calculated
between the vectors).

This implementation is mostly modeled after the streaming_pearson_correlation function in Tensorflow. See
https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3267

This metric delegates to the Covariance metric the tracking of three [co]variances:

- ``covariance(predictions, labels)``, i.e. covariance
- ``covariance(predictions, predictions)``, i.e. variance of ``predictions``
- ``covariance(labels, labels)``, i.e. variance of ``labels``

If we have these values, the sample Pearson correlation coefficient is simply:

r = covariance / (sqrt(predictions_variance) * sqrt(labels_variance))

if predictions_variance or labels_variance is 0, r is 0

### get_metric
```python
PearsonCorrelation.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated sample Pearson correlation.

### reset
```python
PearsonCorrelation.reset(self)
```

Reset any accumulators or internal state.

