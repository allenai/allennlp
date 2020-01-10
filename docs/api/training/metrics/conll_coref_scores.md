# allennlp.training.metrics.conll_coref_scores

## ConllCorefScores
```python
ConllCorefScores(self) -> None
```

### get_metric
```python
ConllCorefScores.get_metric(self, reset:bool=False) -> Tuple[float, float, float]
```

Compute and return the metric. Optionally also call :func:`self.reset`.

### reset
```python
ConllCorefScores.reset(self)
```

Reset any accumulators or internal state.

## Scorer
```python
Scorer(self, metric)
```

Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>

### b_cubed
```python
Scorer.b_cubed(clusters, mention_to_gold)
```

Averaged per-mention precision and recall.
<https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>

### muc
```python
Scorer.muc(clusters, mention_to_gold)
```

Counts the mentions in each predicted cluster which need to be re-allocated in
order for each predicted cluster to be contained by the respective gold cluster.
<https://aclweb.org/anthology/M/M95/M95-1005.pdf>

### phi4
```python
Scorer.phi4(gold_clustering, predicted_clustering)
```

Subroutine for ceafe. Computes the mention F measure between gold and
predicted mentions in a cluster.

### ceafe
```python
Scorer.ceafe(clusters, gold_clusters)
```

Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
Gold and predicted mentions are aligned into clusterings which maximise a metric - in
this case, the F measure between gold and predicted clusters.

<https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>

