# allennlp.models.ensemble

## Ensemble
```python
Ensemble(self, submodels:List[allennlp.models.model.Model]) -> None
```

An ensemble runs multiple instances of a model and selects an answer from the subresults via some
ensembling strategy.

Ensembles differ from most models in that they do not have a vocabulary or weights of their own
(instead they rely on the vocabulary and weights from submodels).  Instead, the submodels are trained
independently and the ensemble is created from the result.

