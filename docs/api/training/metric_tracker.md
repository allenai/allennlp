# allennlp.training.metric_tracker

## MetricTracker
```python
MetricTracker(self, patience:Union[int, NoneType]=None, metric_name:str=None, should_decrease:bool=None) -> None
```

This class tracks a metric during training for the dual purposes of early stopping
and for knowing whether the current value is the best so far. It mimics the PyTorch
`state_dict` / `load_state_dict` interface, so that it can be checkpointed along with
your model and optimizer.

Some metrics improve by increasing; others by decreasing. Here you can either explicitly
supply `should_decrease`, or you can provide a `metric_name` in which case "should decrease"
is inferred from the first character, which must be "+" or "-".

Parameters
----------
patience : int, optional (default = None)
    If provided, then `should_stop_early()` returns True if we go this
    many epochs without seeing a new best value.
metric_name : str, optional (default = None)
    If provided, it's used to infer whether we expect the metric values to
    increase (if it starts with "+") or decrease (if it starts with "-").
    It's an error if it doesn't start with one of those. If it's not provided,
    you should specify ``should_decrease`` instead.
should_decrease : str, optional (default = None)
    If ``metric_name`` isn't provided (in which case we can't infer ``should_decrease``),
    then you have to specify it here.

### clear
```python
MetricTracker.clear(self) -> None
```

Clears out the tracked metrics, but keeps the patience and should_decrease settings.

### state_dict
```python
MetricTracker.state_dict(self) -> Dict[str, Any]
```

A ``Trainer`` can use this to serialize the state of the metric tracker.

### load_state_dict
```python
MetricTracker.load_state_dict(self, state_dict:Dict[str, Any]) -> None
```

A ``Trainer`` can use this to hydrate a metric tracker from a serialized state.

### add_metric
```python
MetricTracker.add_metric(self, metric:float) -> None
```

Record a new value of the metric and update the various things that depend on it.

### add_metrics
```python
MetricTracker.add_metrics(self, metrics:Iterable[float]) -> None
```

Helper to add multiple metrics at once.

### is_best_so_far
```python
MetricTracker.is_best_so_far(self) -> bool
```

Returns true if the most recent value of the metric is the best so far.

### should_stop_early
```python
MetricTracker.should_stop_early(self) -> bool
```

Returns true if improvement has stopped for long enough.

