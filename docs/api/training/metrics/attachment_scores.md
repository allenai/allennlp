# allennlp.training.metrics.attachment_scores

## AttachmentScores
```python
AttachmentScores(self, ignore_classes:List[int]=None) -> None
```

Computes labeled and unlabeled attachment scores for a
dependency parse, as well as sentence level exact match
for both labeled and unlabeled trees. Note that the input
to this metric is the sampled predictions, not the distribution
itself.

Parameters
----------
ignore_classes : ``List[int]``, optional (default = None)
    A list of label ids to ignore when computing metrics.

### get_metric
```python
AttachmentScores.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated metrics as a dictionary.

### reset
```python
AttachmentScores.reset(self)
```

Reset any accumulators or internal state.

