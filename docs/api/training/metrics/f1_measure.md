# allennlp.training.metrics.f1_measure

## F1Measure
```python
F1Measure(self, positive_label:int) -> None
```

Computes Precision, Recall and F1 with respect to a given ``positive_label``.
For example, for a BIO tagging scheme, you would pass the classification index of
the tag you are interested in, resulting in the Precision, Recall and F1 score being
calculated for this tag only.

### get_metric
```python
F1Measure.get_metric(self, reset:bool=False) -> Tuple[float, float, float]
```

Returns
-------
A tuple of the following metrics based on the accumulated count statistics:
precision : float
recall : float
f1-measure : float

