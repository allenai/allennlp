# allennlp.predictors.predictor

## Predictor
```python
Predictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
that can be used for serving models through the web API or making predictions in bulk.

### load_line
```python
Predictor.load_line(self, line:str) -> Dict[str, Any]
```

If your inputs are not in JSON-lines format (e.g. you have a CSV)
you can override this function to parse them correctly.

### dump_line
```python
Predictor.dump_line(self, outputs:Dict[str, Any]) -> str
```

If you don't want your outputs in JSON-lines format
you can override this function to output them differently.

### json_to_labeled_instances
```python
Predictor.json_to_labeled_instances(self, inputs:Dict[str, Any]) -> List[allennlp.data.instance.Instance]
```

Converts incoming json to a :class:`~allennlp.data.instance.Instance`,
runs the model on the newly created instance, and adds labels to the
:class:`~allennlp.data.instance.Instance`s given by the model's output.
Returns
-------
List[instance]
A list of :class:`~allennlp.data.instance.Instance`

### get_gradients
```python
Predictor.get_gradients(self, instances:List[allennlp.data.instance.Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Gets the gradients of the loss with respect to the model inputs.

Parameters
----------
instances: List[Instance]

Returns
-------
Tuple[Dict[str, Any], Dict[str, Any]]
The first item is a Dict of gradient entries for each input.
The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
up to the number of inputs given. The second item is the model's output.

Notes
-----
Takes a ``JsonDict`` representing the inputs of the model and converts
them to :class:`~allennlp.data.instance.Instance`s, sends these through
the model :func:`forward` function after registering hooks on the embedding
layer of the model. Calls :func:`backward` on the loss and then removes the
hooks.

### capture_model_internals
```python
Predictor.capture_model_internals(self) -> Iterator[dict]
```

Context manager that captures the internal-module outputs of
this predictor's model. The idea is that you could use it as follows:

.. code-block:: python

    with predictor.capture_model_internals() as internals:
        outputs = predictor.predict_json(inputs)

    return {**outputs, "model_internals": internals}

### predictions_to_labeled_instances
```python
Predictor.predictions_to_labeled_instances(self, instance:allennlp.data.instance.Instance, outputs:Dict[str, numpy.ndarray]) -> List[allennlp.data.instance.Instance]
```

This function takes a model's outputs for an Instance, and it labels that instance according
to the output. For example, in classification this function labels the instance according
to the class with the highest probability. This function is used to to compute gradients
of what the model predicted. The return type is a list because in some tasks there are
multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
case, each instance in the returned list of Instances contains an individual
entity prediction as the label.

### from_path
```python
Predictor.from_path(archive_path:str, predictor_name:str=None, cuda_device:int=-1, dataset_reader_to_load:str='validation') -> 'Predictor'
```

Instantiate a :class:`Predictor` from an archive path.

If you need more detailed configuration options, such as overrides,
please use `from_archive`.

Parameters
----------
archive_path : ``str``
    The path to the archive.
predictor_name : ``str``, optional (default=None)
    Name that the predictor is registered as, or None to use the
    predictor associated with the model.
cuda_device : ``int``, optional (default=-1)
    If `cuda_device` is >= 0, the model will be loaded onto the
    corresponding GPU. Otherwise it will be loaded onto the CPU.
dataset_reader_to_load : ``str``, optional (default="validation")
    Which dataset reader to load from the archive, either "train" or
    "validation".

Returns
-------
A Predictor instance.

### from_archive
```python
Predictor.from_archive(archive:allennlp.models.archival.Archive, predictor_name:str=None, dataset_reader_to_load:str='validation') -> 'Predictor'
```

Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
that is, from the result of training a model. Optionally specify which `Predictor`
subclass; otherwise, we try to find a corresponding predictor in `DEFAULT_PREDICTORS`, or if
one is not found, the base class (i.e. :class:`Predictor`) will be used. Optionally specify
which :class:`DatasetReader` should be loaded; otherwise, the validation one will be used
if it exists followed by the training dataset reader.

