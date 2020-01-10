# allennlp.models.model

:py:class:`Model` is an abstract class representing
an AllenNLP model.

## Model
```python
Model(self, vocab:allennlp.data.vocabulary.Vocabulary, regularizer:allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator=None) -> None
```

This abstract class represents a model to be trained. Rather than relying completely
on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.

Models built using this API are still compatible with other pytorch models and can
be used naturally as modules within other models - outputs are dictionaries, which
can be unpacked and passed into other layers. One caveat to this is that if you
wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
interleave the models with a wrapper module which unpacks the dictionary into
a list of tensors.

In order for your model to be trained using the :class:`~allennlp.training.Trainer`
api, the output dictionary of your Model must include a "loss" key, which will be
optimised during the training process.

Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
of early stopping and best-model serialization based on a validation metric in
:class:`~allennlp.training.Trainer`. Metrics that begin with "_" will not be logged
to the progress bar by :class:`~allennlp.training.Trainer`.

### get_regularization_penalty
```python
Model.get_regularization_penalty(self) -> Union[float, torch.Tensor]
```

Computes the regularization penalty for the model.
Returns 0 if the model was not configured to use regularization.

### get_parameters_for_histogram_tensorboard_logging
```python
Model.get_parameters_for_histogram_tensorboard_logging(self) -> List[str]
```

Returns the name of model parameters used for logging histograms to tensorboard.

### forward
```python
Model.forward(self, *inputs) -> Dict[str, torch.Tensor]
```

Defines the forward pass of the model. In addition, to facilitate easy training,
this method is designed to compute a loss function defined by a user.

The input is comprised of everything required to perform a
training update, `including` labels - you define the signature here!
It is down to the user to ensure that inference can be performed
without the presence of these labels. Hence, any inputs not available at
inference time should only be used inside a conditional block.

The intended sketch of this method is as follows::

    def forward(self, input1, input2, targets=None):
        ....
        ....
        output1 = self.layer1(input1)
        output2 = self.layer2(input2)
        output_dict = {"output1": output1, "output2": output2}
        if targets is not None:
            # Function returning a scalar torch.Tensor, defined by the user.
            loss = self._compute_loss(output1, output2, targets)
            output_dict["loss"] = loss
        return output_dict

Parameters
----------
inputs:
    Tensors comprising everything needed to perform a training update, `including` labels,
    which should be optional (i.e have a default value of ``None``).  At inference time,
    simply pass the relevant inputs, not including the labels.

Returns
-------
output_dict : ``Dict[str, torch.Tensor]``
    The outputs from the model. In order to train a model using the
    :class:`~allennlp.training.Trainer` api, you must provide a "loss" key pointing to a
    scalar ``torch.Tensor`` representing the loss to be optimized.

### forward_on_instance
```python
Model.forward_on_instance(self, instance:allennlp.data.instance.Instance) -> Dict[str, numpy.ndarray]
```

Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
and returns the result.  Before returning the result, we convert any
``torch.Tensors`` into numpy arrays and remove the batch dimension.

### forward_on_instances
```python
Model.forward_on_instances(self, instances:List[allennlp.data.instance.Instance]) -> List[Dict[str, numpy.ndarray]]
```

Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
arrays using this model's :class:`Vocabulary`, passes those arrays through
:func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
and returns the result.  Before returning the result, we convert any
``torch.Tensors`` into numpy arrays and separate the
batched output into a list of individual dicts per instance. Note that typically
this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
:func:`forward_on_instance`.

Parameters
----------
instances : List[Instance], required
    The instances to run the model on.

Returns
-------
A list of the models output for each instance.

### decode
```python
Model.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Takes the result of :func:`forward` and runs inference / decoding / whatever
post-processing you need to do your model.  The intent is that ``model.forward()`` should
produce potentials or probabilities, and then ``model.decode()`` can take those results and
run some kind of beam search or constrained inference or whatever is necessary.  This does
not handle all possible decoding use cases, but it at least handles simple kinds of
decoding.

This method `modifies` the input dictionary, and also `returns` the same dictionary.

By default in the base class we do nothing.  If your model has some special decoding step,
override this method.

### get_metrics
```python
Model.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

Returns a dictionary of metrics. This method will be called by
:class:`allennlp.training.Trainer` in order to compute and use model metrics for early
stopping and model serialization.  We return an empty dictionary here rather than raising
as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
passed, as frequently a metric accumulator will have some state which should be reset
between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
should be populated during the call to ``forward``, with the
:class:`~allennlp.training.Metric` handling the accumulation of the metric until this
method is called.

### load
```python
Model.load(config:allennlp.common.params.Params, serialization_dir:str, weights_file:str=None, cuda_device:int=-1) -> 'Model'
```

Instantiates an already-trained model, based on the experiment
configuration and some optional overrides.

Parameters
----------
config: Params
    The configuration that was used to train the model. It should definitely
    have a `model` section, and should probably have a `trainer` section
    as well.
serialization_dir: str = None
    The directory containing the serialized weights, parameters, and vocabulary
    of the model.
weights_file: str = None
    By default we load the weights from `best.th` in the serialization
    directory, but you can override that value here.
cuda_device: int = -1
    By default we load the model on the CPU, but if you want to load it
    for GPU usage you can specify the id of your GPU here


Returns
-------
model: Model
    The model specified in the configuration, loaded with the serialized
    vocabulary and the trained weights.

### extend_embedder_vocab
```python
Model.extend_embedder_vocab(self, embedding_sources_mapping:Dict[str, str]=None) -> None
```

Iterates through all embedding modules in the model and assures it can embed
with the extended vocab. This is required in fine-tuning or transfer learning
scenarios where model was trained with original vocabulary but during
fine-tuning/transfer-learning, it will have it work with extended vocabulary
(original + new-data vocabulary).

Parameters
----------
embedding_sources_mapping : Dict[str, str], (optional, default=None)
    Mapping from model_path to pretrained-file path of the embedding
    modules. If pretrained-file used at time of embedding initialization
    isn't available now, user should pass this mapping. Model path is
    path traversing the model attributes upto this embedding module.
    Eg. "_text_field_embedder.token_embedder_tokens".

