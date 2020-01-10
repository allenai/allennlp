# allennlp.nn.initializers

An initializer is just a PyTorch function.
Here we implement a proxy class that allows us
to register them and supply any additional function arguments
(for example, the ``mean`` and ``std`` of a normal initializer)
as named arguments to the constructor.

The available initialization functions are

* `"normal" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.normal_>`_
* `"uniform" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.uniform_>`_
* `"constant" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.constant_>`_
* `"eye" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.eye_>`_
* `"dirac" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.dirac_>`_
* `"xavier_uniform" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_uniform_>`_
* `"xavier_normal" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.xavier_normal_>`_
* `"kaiming_uniform"
  <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_uniform_>`_
* `"kaiming_normal" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.kaiming_normal_>`_
* `"orthogonal" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.orthogonal_>`_
* `"sparse" <https://pytorch.org/docs/master/nn.html?highlight=orthogonal#torch.nn.init.sparse_>`_
* :func:`"block_orthogonal" <block_orthogonal>`
* :func:`"uniform_unit_scaling" <uniform_unit_scaling>`
* :class:`"pretrained" <PretrainedModelInitializer>`

## Initializer
```python
Initializer(self, /, *args, **kwargs)
```

An initializer is really just a bare pytorch function. This class
is a proxy that allows us to implement ``Registerable`` for those functions.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
## uniform_unit_scaling
```python
uniform_unit_scaling(tensor:torch.Tensor, nonlinearity:str='linear')
```

An initaliser which preserves output variance for approximately gaussian
distributed inputs. This boils down to initialising layers using a uniform
distribution in the range ``(-sqrt(3/dim[0]) * scale, sqrt(3 / dim[0]) * scale)``, where
``dim[0]`` is equal to the input dimension of the parameter and the ``scale``
is a constant scaling factor which depends on the non-linearity used.

See `Random Walk Initialisation for Training Very Deep Feedforward Networks
<https://www.semanticscholar.org/paper/Random-Walk-Initialization-for-Training-Very-Deep-Sussillo-Abbott/be9728a0728b6acf7a485225b1e41592176eda0b>`_
for more information.

Parameters
----------
tensor : ``torch.Tensor``, required.
    The tensor to initialise.
nonlinearity : ``str``, optional (default = "linear")
    The non-linearity which is performed after the projection that this
    tensor is involved in. This must be the name of a function contained
    in the ``torch.nn.functional`` package.

Returns
-------
The initialised tensor.

## block_orthogonal
```python
block_orthogonal(tensor:torch.Tensor, split_sizes:List[int], gain:float=1.0) -> None
```

An initializer which allows initializing model parameters in "blocks". This is helpful
in the case of recurrent models which use multiple gates applied to linear projections,
which can be computed efficiently if they are concatenated together. However, they are
separate parameters which should be initialized independently.

Parameters
----------
tensor : ``torch.Tensor``, required.
    A tensor to initialize.
split_sizes : List[int], required.
    A list of length ``tensor.ndim()`` specifying the size of the
    blocks along that particular dimension. E.g. ``[10, 20]`` would
    result in the tensor being split into chunks of size 10 along the
    first dimension and 20 along the second.
gain : float, optional (default = 1.0)
    The gain (scaling) applied to the orthogonal initialization.

## lstm_hidden_bias
```python
lstm_hidden_bias(tensor:torch.Tensor) -> None
```

Initialize the biases of the forget gate to 1, and all other gates to 0,
following Jozefowicz et al., An Empirical Exploration of Recurrent Network Architectures

## PretrainedModelInitializer
```python
PretrainedModelInitializer(self, weights_file_path:str, parameter_name_overrides:Dict[str, str]=None) -> None
```

An initializer which allows initializing parameters using a pretrained model. The
initializer will load all of the weights from the ``weights_file_path`` and use the
name of the new parameters to index into the pretrained parameters. Therefore,
by default, the names of the new and pretrained parameters must be the same.
However, this behavior can be overridden using the ``parameter_name_overrides``,
which remaps the name of the new parameter to the key which should be used
to index into the pretrained parameters.

The initializer will load all of the weights from the ``weights_file_path``
regardless of which parameters will actually be used to initialize the new model.
So, if you need to initialize several parameters using a pretrained model, the most
memory-efficient way to do this is to use one ``PretrainedModelInitializer`` per
weights file and use a regex to match all of the new parameters which need to be
initialized.

The below entry in the :class:`InitializerApplicator` parameters will initialize
``linear_1.weight`` and ``linear_2.weight`` using a pretrained model.
``linear_1.weight`` will be initialized to the pretrained
parameters called ``linear_1.weight``, but ``linear_2.weight`` will be initialized
to the pretrained parameters called ``linear_3.weight``::

   ["linear_1.weight|linear_2.weight",
       {
           "type": "pretrained",
           "weights_file_path": "best.th",
           "parameter_name_overrides": {
               "linear_2.weight": "linear_3.weight"
           }
       }
   ]

To initialize weights for all the parameters from a pretrained model (assuming their names
remain unchanged), use the following instead:

    .. code-block:: js

        [".*",
            {
                "type": "pretrained",
                "weights_file_path": "best.th",
                "parameter_name_overrides": {}
            }
        ]

Parameters
----------
weights_file_path : ``str``, required
    The path to the weights file which has the pretrained model parameters.
parameter_name_overrides : ``Dict[str, str]``, optional (default = None)
    The mapping from the new parameter name to the name which should be used
    to index into the pretrained model parameters. If a parameter name is not
    specified, the initializer will use the parameter's default name as the key.

## InitializerApplicator
```python
InitializerApplicator(self, initializers:List[Tuple[str, allennlp.nn.initializers.Initializer]]=None, prevent_regexes:List[str]=None) -> None
```

Applies initializers to the parameters of a Module based on regex matches.  Any parameter not
explicitly matching a regex will not be initialized, instead using whatever the default
initialization was in the module's code.

### from_params
```python
InitializerApplicator.from_params(params:List[Tuple[str, allennlp.common.params.Params]]=None) -> 'InitializerApplicator'
```

Converts a Params object into an InitializerApplicator. The json should
be formatted as follows::

    [
        ["parameter_regex_match1",
            {
                "type": "normal"
                "mean": 0.01
                "std": 0.1
            }
        ],
        ["parameter_regex_match2", "uniform"]
        ["prevent_init_regex", "prevent"]
    ]

where the first item in each tuple is the regex that matches to parameters, and the second
item is a set of parameters that will be passed to ``Initialzer.from_params()``.  These
values can either be strings, in which case they correspond to the names of initializers,
or dictionaries, in which case they must contain the "type" key, corresponding to the name
of an initializer.  In addition, they may contain auxiliary named parameters which will be
fed to the initializer itself. To determine valid auxiliary parameters, please refer to the
torch.nn.init documentation. Only "prevent" is a special type which does not have corresponding
initializer. Any parameter matching its corresponding regex will be overridden to NOT initialize.

Returns
-------
An InitializerApplicator containing the specified initializers.

