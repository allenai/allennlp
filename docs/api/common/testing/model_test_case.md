# allennlp.common.testing.model_test_case

## ModelTestCase
```python
ModelTestCase(self, methodName='runTest')
```

A subclass of :class:`~allennlp.common.testing.test_case.AllenNlpTestCase`
with added methods for testing :class:`~allennlp.models.model.Model` subclasses.

### ensure_model_can_train_save_and_load
```python
ModelTestCase.ensure_model_can_train_save_and_load(self, param_file:str, tolerance:float=0.0001, cuda_device:int=-1, gradients_to_ignore:Set[str]=None, overrides:str='', disable_dropout:bool=True)
```

Parameters
----------
param_file : ``str``
    Path to a training configuration file that we will use to train the model for this
    test.
tolerance : ``float``, optional (default=1e-4)
    When comparing model predictions between the originally-trained model and the model
    after saving and loading, we will use this tolerance value (passed as ``rtol`` to
    ``numpy.testing.assert_allclose``).
cuda_device : ``int``, optional (default=-1)
    The device to run the test on.
gradients_to_ignore : ``Set[str]``, optional (default=None)
    This test runs a gradient check to make sure that we're actually computing gradients
    for all of the parameters in the model.  If you really want to ignore certain
    parameters when doing that check, you can pass their names here.  This is not
    recommended unless you're `really` sure you don't need to have non-zero gradients for
    those parameters (e.g., some of the beam search / state machine models have
    infrequently-used parameters that are hard to force the model to use in a small test).
overrides : ``str``, optional (default = "")
    A JSON string that we will use to override values in the input parameter file.
disable_dropout : ``bool``, optional (default = True)
    If True we will set all dropout to 0 before checking gradients. (Otherwise, with small
    datasets, you may get zero gradients because of unlucky dropout.)

### ensure_batch_predictions_are_consistent
```python
ModelTestCase.ensure_batch_predictions_are_consistent(self, keys_to_ignore:Iterable[str]=())
```

Ensures that the model performs the same on a batch of instances as on individual instances.
Ignores metrics matching the regexp .*loss.* and those specified explicitly.

Parameters
----------
keys_to_ignore : ``Iterable[str]``, optional (default=())
    Names of metrics that should not be taken into account, e.g. "batch_weight".

