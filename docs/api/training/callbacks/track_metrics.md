# allennlp.training.callbacks.track_metrics

## TrackMetrics
```python
TrackMetrics(self, patience:int=None, validation_metric:str='-loss') -> None
```

Callback that handles tracking of metrics and (potentially) early stopping.

Parameters
----------
patience : int, optional (default = None)
    If a positive number is provided, training will stop when the supplied
    validation_metric has not improved in this many epochs.
validation_metric : str, optional (default = "-loss")
    The metric to use for early stopping. The initial +/- indicates whether
    we expect the metric to increase or decrease during training.

