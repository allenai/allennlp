# allennlp.training.metrics.srl_eval_scorer

## SrlEvalScorer
```python
SrlEvalScorer(self, srl_eval_path:str='/Users/markn/allen_ai/allennlp/allennlp/tools/srl-eval.pl', ignore_classes:List[str]=None) -> None
```

This class uses the external srl-eval.pl script for computing the CoNLL SRL metrics.

AllenNLP contains the srl-eval.pl script, but you will need perl 5.x.

Note that this metric reads and writes from disk quite a bit. In particular, it
writes and subsequently reads two files per __call__, which is typically invoked
once per batch. You probably don't want to include it in your training loop;
instead, you should calculate this on a validation set only.

Parameters
----------
srl_eval_path : ``str``, optional.
    The path to the srl-eval.pl script.
ignore_classes : ``List[str]``, optional (default=``None``).
    A list of classes to ignore.

### get_metric
```python
SrlEvalScorer.get_metric(self, reset:bool=False)
```

Returns
-------
A Dict per label containing following the span based metrics:
precision : float
recall : float
f1-measure : float

Additionally, an ``overall`` key is included, which provides the precision,
recall and f1-measure for all spans.

