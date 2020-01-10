# allennlp.training.trainer_pieces

## TrainerPieces
```python
TrainerPieces(self, /, *args, **kwargs)
```

We would like to avoid having complex instantiation logic taking place
in `Trainer.from_params`. This helper class has a `from_params` that
instantiates a model, loads train (and possibly validation and test) datasets,
constructs a Vocabulary, creates data iterators, and handles a little bit
of bookkeeping. If you're creating your own alternative training regime
you might be able to use this.

### iterator
Alias for field number 1
### model
Alias for field number 0
### params
Alias for field number 6
### test_dataset
Alias for field number 4
### train_dataset
Alias for field number 2
### validation_dataset
Alias for field number 3
### validation_iterator
Alias for field number 5
