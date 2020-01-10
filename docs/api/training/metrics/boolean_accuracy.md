# allennlp.training.metrics.boolean_accuracy

## BooleanAccuracy
```python
BooleanAccuracy(self) -> None
```

Just checks batch-equality of two tensors and computes an accuracy metric based on that.
That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
that are totally masked are ignored (in which case the denominator is the number of predictions that have
at least one unmasked element).

This is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
predictions.  If you have categorical output, though, you should typically just use
:class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
some kind of constrained inference and don't have a prediction tensor that matches the API of
:class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.

### get_metric
```python
BooleanAccuracy.get_metric(self, reset:bool=False)
```

Returns
-------
The accumulated accuracy.

### reset
```python
BooleanAccuracy.reset(self)
```

Reset any accumulators or internal state.

