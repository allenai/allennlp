# allennlp.training.metrics.bleu

## BLEU
```python
BLEU(self, ngram_weights:Iterable[float]=(0.25, 0.25, 0.25, 0.25), exclude_indices:Set[int]=None) -> None
```

Bilingual Evaluation Understudy (BLEU).

BLEU is a common metric used for evaluating the quality of machine translations
against a set of reference translations. See `Papineni et. al.,
"BLEU: a method for automatic evaluation of machine translation", 2002
<https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/>`_.

Parameters
----------
ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
    Weights to assign to scores for each ngram size.
exclude_indices : ``Set[int]``, optional (default = None)
    Indices to exclude when calculating ngrams. This should usually include
    the indices of the start, end, and pad tokens.

Notes
-----
We chose to implement this from scratch instead of wrapping an existing implementation
(such as `nltk.translate.bleu_score`) for a two reasons. First, so that we could
pass tensors directly to this metric instead of first converting the tensors to lists of strings.
And second, because functions like `nltk.translate.bleu_score.corpus_bleu()` are
meant to be called once over the entire corpus, whereas it is more efficient
in our use case to update the running precision counts every batch.

This implementation only considers a reference set of size 1, i.e. a single
gold target sequence for each predicted sequence.

### reset
```python
BLEU.reset(self) -> None
```

Reset any accumulators or internal state.

### get_metric
```python
BLEU.get_metric(self, reset:bool=False) -> Dict[str, float]
```

Compute and return the metric. Optionally also call :func:`self.reset`.

