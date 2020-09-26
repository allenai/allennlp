"""
The `test-install` subcommand provides a programmatic way to verify
that AllenNLP has been successfully installed.
"""

import argparse
import logging
import pathlib

from overrides import overrides
import torch

import allennlp
from allennlp.common.util import import_module_and_submodules
from allennlp.commands.subcommand import Subcommand
from allennlp.version import VERSION


logger = logging.getLogger(__name__)


@Subcommand.register("test-install")
class TestInstall(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Test that AllenNLP is installed correctly."""
        subparser = parser.add_parser(
            self.name, description=description, help="Test AllenNLP installation."
        )
        subparser.set_defaults(func=_run_test)
        return subparser


def _get_module_root():
    return pathlib.Path(allennlp.__file__).parent


def _run_test(args: argparse.Namespace):
    # Make sure we can actually import the main modules without errors.
    import_module_and_submodules("allennlp.common")
    import_module_and_submodules("allennlp.data")
    import_module_and_submodules("allennlp.interpret")
    import_module_and_submodules("allennlp.models")
    import_module_and_submodules("allennlp.modules")
    import_module_and_submodules("allennlp.nn")
    import_module_and_submodules("allennlp.predictors")
    import_module_and_submodules("allennlp.training")
    logger.info("AllenNLP version %s installed to %s", VERSION, _get_module_root())
    logger.info("Cuda devices available: %s", torch.cuda.device_count())
