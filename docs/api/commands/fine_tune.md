# allennlp.commands.fine_tune

The ``fine-tune`` subcommand is used to continue training (or `fine-tune`) a model on a `different
dataset` than the one it was originally trained on.  It requires a saved model archive file, a path
to the data you will continue training with, and a directory in which to write the results.

.. code-block:: bash

   $ allennlp fine-tune --help
    usage: allennlp fine-tune [-h] -m MODEL_ARCHIVE -c CONFIG_FILE -s
                              SERIALIZATION_DIR [-o OVERRIDES] [--extend-vocab]
                              [--file-friendly-logging]
                              [--batch-weight-key BATCH_WEIGHT_KEY]
                              [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                              [--include-package INCLUDE_PACKAGE]

    Continues training a saved model on a new dataset.

    optional arguments:
      -h, --help            show this help message and exit
      -m MODEL_ARCHIVE, --model-archive MODEL_ARCHIVE
                            path to the saved model archive from training on the
                            original data
      -c CONFIG_FILE, --config-file CONFIG_FILE
                            configuration file to use for training. Format is the
                            same as for the "train" command, but the "model"
                            section is ignored.
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the fine-tuned model and
                            its logs
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the training
                            configuration (only affects the config_file, _not_ the
                            model_archive)
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## fine_tune_model_from_args
```python
fine_tune_model_from_args(args:argparse.Namespace)
```

Just converts from an ``argparse.Namespace`` object to string paths.

## fine_tune_model_from_file_paths
```python
fine_tune_model_from_file_paths(model_archive_path:str, config_file:str, serialization_dir:str, overrides:str='', extend_vocab:bool=False, file_friendly_logging:bool=False, batch_weight_key:str='', embedding_sources_mapping:str='') -> allennlp.models.model.Model
```

A wrapper around :func:`fine_tune_model` which loads the model archive from a file.

Parameters
----------
model_archive_path : ``str``
    Path to a saved model archive that is the result of running the ``train`` command.
config_file : ``str``
    A configuration file specifying how to continue training.  The format is identical to the
    configuration file for the ``train`` command, but any contents in the ``model`` section is
    ignored (as we are using the provided model archive instead).
serialization_dir : ``str``
    The directory in which to save results and logs. We just pass this along to
    :func:`fine_tune_model`.
overrides : ``str``
    A JSON string that we will use to override values in the input parameter file.
extend_vocab : ``bool``, optional (default=False)
    If ``True``, we use the new instances to extend your vocabulary.
file_friendly_logging : ``bool``, optional (default=False)
    If ``True``, we make our output more friendly to saved model files.  We just pass this
    along to :func:`fine_tune_model`.
batch_weight_key : ``str``, optional (default="")
    If non-empty, name of metric used to weight the loss on a per-batch basis.
embedding_sources_mapping : ``str``, optional (default="")
    JSON string to define dict mapping from embedding paths used during training to
    the corresponding embedding filepaths available during fine-tuning.

## fine_tune_model
```python
fine_tune_model(model:allennlp.models.model.Model, params:allennlp.common.params.Params, serialization_dir:str, extend_vocab:bool=False, file_friendly_logging:bool=False, batch_weight_key:str='', embedding_sources_mapping:Dict[str, str]=None) -> allennlp.models.model.Model
```

Fine tunes the given model, using a set of parameters that is largely identical to those used
for :func:`~allennlp.commands.train.train_model`, except that the ``model`` section is ignored,
if it is present (as we are already given a ``Model`` here).

The main difference between the logic done here and the logic done in ``train_model`` is that
here we do not worry about vocabulary construction or creating the model object.  Everything
else is the same.

Parameters
----------
model : ``Model``
    A model to fine tune.
params : ``Params``
    A parameter object specifying an AllenNLP Experiment
serialization_dir : ``str``
    The directory in which to save results and logs.
extend_vocab : ``bool``, optional (default=False)
    If ``True``, we use the new instances to extend your vocabulary.
file_friendly_logging : ``bool``, optional (default=False)
    If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
    down tqdm's output to only once every 10 seconds.
batch_weight_key : ``str``, optional (default="")
    If non-empty, name of metric used to weight the loss on a per-batch basis.
embedding_sources_mapping : ``Dict[str, str]``, optional (default=None)
    mapping from model paths to the pretrained embedding filepaths
    used during fine-tuning.

