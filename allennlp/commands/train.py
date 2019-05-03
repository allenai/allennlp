"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help

   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                         [--file-friendly-logging]
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
     --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

import argparse
import logging
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, cleanup_global_logging, dump_metrics
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.util import create_serialization_dir, evaluate

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model.')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-f', '--force',
                               action='store_true',
                               required=False,
                               help='overwrite the output directory if it exists')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.add_argument('--cache-directory',
                               type=str,
                               default='',
                               help='Location to store cache of data preprocessing')

        subparser.add_argument('--cache-prefix',
                               type=str,
                               default='',
                               help='Prefix to use for data caching, giving current parameter '
                               'settings a name in the cache, instead of computing a hash')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover,
                          args.force,
                          args.cache_directory,
                          args.cache_prefix)


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          cache_directory: str = None,
                          cache_prefix: str = None) -> Model:
    """
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
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params,
                       serialization_dir,
                       file_friendly_logging,
                       recover,
                       force,
                       cache_directory, cache_prefix)


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False,
                cache_directory: str = None,
                cache_prefix: str = None) -> Model:
    """
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

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)
    create_serialization_dir(params, serialization_dir, recover, force)
    stdout_handler = prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == "default":
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(params,  # pylint: disable=no-member
                                           serialization_dir,
                                           recover,
                                           cache_directory,
                                           cache_prefix)
        trainer = Trainer.from_params(
                model=pieces.model,
                serialization_dir=serialization_dir,
                iterator=pieces.iterator,
                train_data=pieces.train_dataset,
                validation_data=pieces.validation_dataset,
                params=pieces.params,
                validation_iterator=pieces.validation_iterator)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset

    else:
        trainer = TrainerBase.from_params(params, serialization_dir, recover)
        # TODO(joelgrus): handle evaluation in the general case
        evaluation_iterator = evaluation_dataset = None

    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                                batch_weight_key="")

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    cleanup_global_logging(stdout_handler)

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model
