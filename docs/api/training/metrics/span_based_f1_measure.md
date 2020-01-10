# allennlp.training.metrics.span_based_f1_measure

## SpanBasedF1Measure
```python
SpanBasedF1Measure(self, vocabulary:allennlp.data.vocabulary.Vocabulary, tag_namespace:str='tags', ignore_classes:List[str]=None, label_encoding:Union[str, NoneType]='BIO', tags_to_spans_function:Union[Callable[[List[str], Union[List[str], NoneType]], List[Tuple[str, Tuple[int, int]]]], NoneType]=None) -> None
```

The Conll SRL metrics are based on exact span matching. This metric
implements span-based precision and recall metrics for a BIO tagging
scheme. It will produce precision, recall and F1 measures per tag, as
well as overall statistics. Note that the implementation of this metric
is not exactly the same as the perl script used to evaluate the CONLL 2005
data - particularly, it does not consider continuations or reference spans
as constituents of the original span. However, it is a close proxy, which
can be helpful for judging model performance during training. This metric
works properly when the spans are unlabeled (i.e., your labels are
simply "B", "I", "O" if using the "BIO" label encoding).


### get_metric
```python
SpanBasedF1Measure.get_metric(self, reset:bool=False)
```

Returns
-------
A Dict per label containing following the span based metrics:
precision : float
recall : float
f1-measure : float

Additionally, an ``overall`` key is included, which provides the precision,
recall and f1-measure for all spans.

