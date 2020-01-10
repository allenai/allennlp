# allennlp.training.metrics.fbeta_measure

## FBetaMeasure
```python
FBetaMeasure(self, beta:float=1.0, average:str=None, labels:List[int]=None) -> None
```
Compute precision, recall, F-measure and support for each class.

The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
true positives and ``fp`` the number of false positives. The precision is
intuitively the ability of the classifier not to label as positive a sample
that is negative.

The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
true positives and ``fn`` the number of false negatives. The recall is
intuitively the ability of the classifier to find all the positive samples.

The F-beta score can be interpreted as a weighted harmonic mean of
the precision and recall, where an F-beta score reaches its best
value at 1 and worst score at 0.

If we have precision and recall, the F-beta score is simply:
``F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)``

The F-beta score weights recall more than precision by a factor of
``beta``. ``beta == 1.0`` means recall and precision are equally important.

The support is the number of occurrences of each class in ``y_true``.

Parameters
----------
beta : ``float``, optional (default = 1.0)
    The strength of recall versus precision in the F-score.

average : string, [None (default), 'micro', 'macro']
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted mean.
        This does not take label imbalance into account.

labels: list, optional
    The set of labels to include and their order if ``average is None``.
    Labels present in the data can be excluded, for example to calculate a
    multi-class average ignoring a majority negative class. Labels not present
    in the data will result in 0 components in a macro average.


### get_metric
```python
FBetaMeasure.get_metric(self, reset:bool=False)
```

Returns
-------
A tuple of the following metrics based on the accumulated count statistics:
precisions : List[float]
recalls : List[float]
f1-measures : List[float]

If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.

### reset
```python
FBetaMeasure.reset(self) -> None
```

Reset any accumulators or internal state.

