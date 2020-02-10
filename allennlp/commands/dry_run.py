"""
The ``dry-run`` command creates a vocabulary, informs you of
dataset statistics and other training utilities without actually training
a model.

.. code-block:: bash

    $ allennlp dry-run --help
    usage: allennlp dry-run [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                            [--include-package INCLUDE_PACKAGE]
                            param_path

    Create a vocabulary, compute dataset statistics and other training utilities.

    positional arguments:
      param_path            path to parameter file describing the model and its
                            inputs

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the output of the dry run.
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.models import Model
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


@Subcommand.register("dry-run")
class DryRun(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = (
            """Create a vocabulary, compute dataset statistics and other training utilities."""
        )
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Create a vocabulary, compute dataset statistics and other training utilities.",
        )
        subparser.add_argument(
            "param_path",
            type=str,
            help="path to parameter file describing the model and its inputs",
        )
        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the output of the dry run.",
        )
        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.set_defaults(func=dry_run_from_args)

        return subparser


def dry_run_from_args(args: argparse.Namespace) -> None:
    """
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    serialization_dir = args.serialization_dir
    overrides_ = args.overrides

    params = Params.from_file(parameter_path, overrides_)

    dry_run_from_params(params, serialization_dir)


def dry_run_from_params(params: Params, serialization_dir: str) -> None:
    prepare_environment(params)

    vocab = training_util.make_vocab_from_params(params, serialization_dir)

    # We create the model so the logging info is shown.
    Model.from_params(vocab=vocab, params=params.pop("model"))
