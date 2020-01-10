# allennlp.commands.find_learning_rate

The ``find-lr`` subcommand can be used to find a good learning rate for a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp find-lr --help
    usage: allennlp find-lr [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                            [--start-lr START_LR] [--end-lr END_LR]
                            [--num-batches NUM_BATCHES]
                            [--stopping-factor STOPPING_FACTOR] [--linear] [-f]
                            [--include-package INCLUDE_PACKAGE]
                            param_path

    Find a learning rate range where loss decreases quickly for the specified
    model and dataset.

    positional arguments:
      param_path            path to parameter file describing the model to be
                            trained

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            The directory in which to save results.
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --start-lr START_LR   learning rate to start the search (default = 1e-05)
      --end-lr END_LR       learning rate up to which search is done (default =
                            10)
      --num-batches NUM_BATCHES
                            number of mini-batches to run learning rate finder
                            (default = 100)
      --stopping-factor STOPPING_FACTOR
                            stop the search when the current loss exceeds the best
                            loss recorded by multiple of stopping factor
      --linear              increase learning rate linearly instead of exponential
                            increase
      -f, --force           overwrite the output directory if it exists
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## find_learning_rate_from_args
```python
find_learning_rate_from_args(args:argparse.Namespace) -> None
```

Start learning rate finder for given args

## find_learning_rate_model
```python
find_learning_rate_model(params:allennlp.common.params.Params, serialization_dir:str, start_lr:float=1e-05, end_lr:float=10, num_batches:int=100, linear_steps:bool=False, stopping_factor:float=None, force:bool=False) -> None
```

Runs learning rate search for given `num_batches` and saves the results in ``serialization_dir``

Parameters
----------
params : ``Params``
    A parameter object specifying an AllenNLP Experiment.
serialization_dir : ``str``
    The directory in which to save results.
start_lr : ``float``
    Learning rate to start the search.
end_lr : ``float``
    Learning rate upto which search is done.
num_batches : ``int``
    Number of mini-batches to run Learning rate finder.
linear_steps : ``bool``
    Increase learning rate linearly if False exponentially.
stopping_factor : ``float``
    Stop the search when the current loss exceeds the best loss recorded by
    multiple of stopping factor. If ``None`` search proceeds till the ``end_lr``
force : ``bool``
    If True and the serialization directory already exists, everything in it will
    be removed prior to finding the learning rate.

## search_learning_rate
```python
search_learning_rate(trainer:allennlp.training.trainer.Trainer, start_lr:float=1e-05, end_lr:float=10, num_batches:int=100, linear_steps:bool=False, stopping_factor:float=None) -> Tuple[List[float], List[float]]
```

Runs training loop on the model using :class:`~allennlp.training.trainer.Trainer`
increasing learning rate from ``start_lr`` to ``end_lr`` recording the losses.
Parameters
----------
trainer: :class:`~allennlp.training.trainer.Trainer`
start_lr : ``float``
    The learning rate to start the search.
end_lr : ``float``
    The learning rate upto which search is done.
num_batches : ``int``
    Number of batches to run the learning rate finder.
linear_steps : ``bool``
    Increase learning rate linearly if False exponentially.
stopping_factor : ``float``
    Stop the search when the current loss exceeds the best loss recorded by
    multiple of stopping factor. If ``None`` search proceeds till the ``end_lr``
Returns
-------
(learning_rates, losses) : ``Tuple[List[float], List[float]]``
    Returns list of learning rates and corresponding losses.
    Note: The losses are recorded before applying the corresponding learning rate

