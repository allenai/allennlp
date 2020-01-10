# allennlp.training.callbacks.validate

## Validate
```python
Validate(self, validation_data:Iterable[allennlp.data.instance.Instance], validation_iterator:allennlp.data.iterators.data_iterator.DataIterator) -> None
```

Evaluates the trainer's ``Model`` using the provided validation dataset.
Uses the results to populate trainer.val_metrics.

Parameters
----------
validation_data : ``Iterable[Instance]``
    The instances in the validation dataset.
validation_iterator : ``DataIterator``
    The iterator to use in the evaluation.

