# allennlp.commands.train

The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help
    usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                          [--file-friendly-logging]
                          [--cache-directory CACHE_DIRECTORY]
                          [--cache-prefix CACHE_PREFIX] [--node-rank NODE_RANK]
                          [--include-package INCLUDE_PACKAGE]
                          param_path

    Train the specified model on the specified dataset.

    positional arguments:
      param_path            path to parameter file describing the model to be
                            trained

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
      -r, --recover         recover training from the state in serialization_dir
      -f, --force           overwrite the output directory if it exists
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate
      --cache-directory CACHE_DIRECTORY
                            Location to store cache of data preprocessing
      --cache-prefix CACHE_PREFIX
                            Prefix to use for data caching, giving current
                            parameter settings a name in the cache, instead of
                            computing a hash
      --node-rank NODE_RANK
                            Rank of this node in the distributed setup (default =
                            0)
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## train_model_from_args
```python
train_model_from_args(args:argparse.Namespace)
```

Just converts from an ``argparse.Namespace`` object to string paths.

## train_model_from_file
```python
train_model_from_file(parameter_filename:str, serialization_dir:str, overrides:str='', file_friendly_logging:bool=False, recover:bool=False, force:bool=False, cache_directory:str=None, cache_prefix:str=None, node_rank:int=0, include_package:List[str]=None) -> allennlp.models.model.Model
```

A wrapper around :func:`train_model` which loads the params from a file.

Parameters
----------
parameter_filename : ``str``
    A json parameter file specifying an AllenNLP experiment.
serialization_dir : ``str``
    The directory in which to save results and logs. We just pass this along to
    :func:`train_model`.
overrides : ``str``
    A JSON string that we will use to override values in the input parameter file.
file_friendly_logging : ``bool``, optional (default=False)
    If ``True``, we make our output more friendly to saved model files.  We just pass this
    along to :func:`train_model`.
recover : ``bool`, optional (default=False)
    If ``True``, we will try to recover a training run from an existing serialization
    directory.  This is only intended for use when something actually crashed during the middle
    of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
force : ``bool``, optional (default=False)
    If ``True``, we will overwrite the serialization directory if it already exists.
cache_directory : ``str``, optional
    For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
cache_prefix : ``str``, optional
    For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
node_rank : ``int``, optional
    Rank of the current node in distributed training
include_package : ``str``, optional
    In distributed mode, extra packages mentioned will be imported in trainer workers.

## train_model
```python
train_model(params:allennlp.common.params.Params, serialization_dir:str, file_friendly_logging:bool=False, recover:bool=False, force:bool=False, cache_directory:str=None, cache_prefix:str=None, node_rank:int=0, include_package:List[str]=None) -> allennlp.models.model.Model
```

Trains the model specified in the given :class:`Params` object, using the data and training
parameters also specified in that object, and saves the results in ``serialization_dir``.

Parameters
----------
params : ``Params``
    A parameter object specifying an AllenNLP Experiment.
serialization_dir : ``str``
    The directory in which to save results and logs.
file_friendly_logging : ``bool``, optional (default=False)
    If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
    down tqdm's output to only once every 10 seconds.
recover : ``bool``, optional (default=False)
    If ``True``, we will try to recover a training run from an existing serialization
    directory.  This is only intended for use when something actually crashed during the middle
    of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
force : ``bool``, optional (default=False)
    If ``True``, we will overwrite the serialization directory if it already exists.
cache_directory : ``str``, optional
    For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
cache_prefix : ``str``, optional
    For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
node_rank : ``int``, optional
    Rank of the current node in distributed training
include_package : ``List[str]``, optional
    In distributed mode, extra packages mentioned will be imported in trainer workers.

Returns
-------
best_model : ``Model``
    The model with the best epoch weights.

