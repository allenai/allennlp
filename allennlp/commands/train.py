"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: python -m allennlp.run [command] train [-h] -s SERIALIZATION_DIR
                                               [-o OVERRIDES]
                                               [--include-package INCLUDE_PACKAGE]
                                               [--file-friendly-logging]
                                               param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
   -h, --help            show this help message and exit
   -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
   -o OVERRIDES, --overrides OVERRIDES
                           a HOCON structure used to override the experiment
                           configuration
   --include-package INCLUDE_PACKAGE
                           additional packages to include
   --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
"""
from typing import Dict, Iterable
import argparse
import json
import logging
import os
import sys
from copy import deepcopy

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(
                name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        # This is necessary to preserve backward compatibility
        serialization = subparser.add_mutually_exclusive_group(required=True)
        serialization.add_argument('-s', '--serialization-dir',
                                   type=str,
                                   help='directory in which to save the model and its logs')
        serialization.add_argument('--serialization_dir',
                                   type=str,
                                   help=argparse.SUPPRESS)

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    # Import any additional modules needed (to register custom classes)
    for package_name in args.include_package:
        import_submodules(package_name)
    train_model_from_file(args.param_path, args.serialization_dir, args.overrides, args.file_friendly_logging)


def train_model_from_file(parameter_filename: str, serialization_dir: str, overrides: str = "",
                          file_friendly_logging: bool = False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    if file_friendly_logging:
        Tqdm.set_default_mininterval(10.0)
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets

def train_model(params: Params, serialization_dir: str, file_friendly_logging: bool = False) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)

    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), # type: ignore
                           sys.stdout, file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr, file_friendly_logging)
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   (instance for key, dataset in all_datasets.items()
                                    for instance in dataset
                                    if key in datasets_for_vocab_creation))
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')
    metrics = trainer.train()

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    if test_data and evaluate_on_test:
        test_metrics = evaluate(model, test_data, iterator, cuda_device=trainer._cuda_devices[0])  # pylint: disable=protected-access
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    return model
