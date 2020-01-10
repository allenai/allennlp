# allennlp.training.metrics.perplexity

## Perplexity
```python
Perplexity(self) -> None
```

Perplexity is a common metric used for evaluating how well a language model
predicts a sample.

Notes
-----
Assumes negative log likelihood loss of each batch (base e). Provides the
average perplexity of the batches.

### get_metric
```python
Perplexity.get_metric(self, reset:bool=False) -> float
```

Returns
-------
The accumulated perplexity.

