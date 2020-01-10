# allennlp.training.metrics.metric

## Metric
```python
Metric(self, /, *args, **kwargs)
```

A very general abstract class representing a metric which can be
accumulated.

### get_metric
```python
Metric.get_metric(self, reset:bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]
```

Compute and return the metric. Optionally also call :func:`self.reset`.

### reset
```python
Metric.reset(self) -> None
```

Reset any accumulators or internal state.

### unwrap_to_tensors
```python
Metric.unwrap_to_tensors(*tensors:torch.Tensor)
```

If you actually passed gradient-tracking Tensors to a Metric, there will be
a huge memory leak, because it will prevent garbage collection for the computation
graph. This method ensures that you're using tensors directly and that they are on
the CPU.

