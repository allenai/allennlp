"""
The ``make-vocab`` subcommand allows you to create a vocabulary from
your dataset[s], which you can then reuse without recomputing it
each training run.

.. code-block:: bash

   $ python -m allennlp.run make-vocab --help

    usage: python -m allennlp.run make-vocab [-h] [-o OVERRIDES] param_path

    Create a vocabulary from the specified dataset.

    positional arguments:
    param_path            path to parameter file describing the model and its
                          inputs

    optional arguments:
    -h, --help            show this help message and exit
    -o OVERRIDES, --overrides OVERRIDES
                          a HOCON structure used to override the experiment
                          configuration
"""
import argparse
import logging
import os

from allennlp.commands.train import datasets_from_params
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MakeVocab(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Create a vocabulary from the specified dataset.'''
        subparser = parser.add_parser(
                name, description=description, help='Create a vocabulary')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model and its inputs')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=make_vocab_from_args)

        return subparser


def make_vocab_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    overrides = args.overrides

    params = Params.from_file(parameter_path, overrides)

    make_vocab_from_params(params)

def make_vocab_from_params(params: Params):
    prepare_environment(params)

    vocab_params = params.pop("vocabulary", {})
    vocab_dir = vocab_params.get('directory_path')
    if vocab_dir is None:
        raise ConfigurationError("To use `make-vocab` your configuration must contain a value "
                                 "at vocabulary.directory_path")

    os.makedirs(vocab_dir, exist_ok=True)

    all_datasets = datasets_from_params(params)

    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(Params({}),
                                   (instance for key, dataset in all_datasets.items()
                                    for instance in dataset
                                    if key in datasets_for_vocab_creation))

    vocab.save_to_files(vocab_dir)
    logger.info("done creating vocab")
