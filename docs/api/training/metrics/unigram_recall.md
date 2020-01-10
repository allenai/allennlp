# allennlp.training.metrics.unigram_recall

## UnigramRecall
```python
UnigramRecall(self) -> None
```

Unigram top-K recall. This does not take word order into account. Assumes
integer labels, with each item to be classified having a single correct
class.

### get_metric
```python
UnigramRecall.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated recall.

### reset
```python
UnigramRecall.reset(self)
```

Reset any accumulators or internal state.

