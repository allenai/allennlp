"""
The ``test-install`` subcommand verifies
an installation by running the unit tests.

    $ allennlp test-install --help
    usage: allennlp test-install [-h] [--run-all] [-k K]
                                 [--include-package INCLUDE_PACKAGE]

    Test that installation works by running the unit tests.

    optional arguments:
      -h, --help            show this help message and exit
      --run-all             By default, we skip tests that are slow or download
                            large files. This flag will run all tests.
      -k K                  Limit tests by setting pytest -k argument
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

import argparse
import logging
import os
import pathlib
import sys

import pytest
from overrides import overrides

import allennlp
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import pushd

logger = logging.getLogger(__name__)


@Subcommand.register("test-install")
class TestInstall(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:

        description = """Test that installation works by running the unit tests."""
        subparser = parser.add_parser(
            self.name, description=description, help="Run the unit tests."
        )

        subparser.add_argument(
            "--run-all",
            action="store_true",
            help="By default, we skip tests that are slow "
            "or download large files. This flag will run all tests.",
        )
        subparser.add_argument(
            "-k", type=str, default=None, help="Limit tests by setting pytest -k argument"
        )

        subparser.set_defaults(func=_run_test)

        return subparser


def _get_module_root():
    return pathlib.Path(allennlp.__file__).parent


def _run_test(args: argparse.Namespace):
    module_parent = _get_module_root().parent
    logger.info("Changing directory to %s", module_parent)
    with pushd(module_parent):
        test_dir = os.path.join(module_parent, "allennlp")
        logger.info("Running tests at %s", test_dir)

        if args.k:
            pytest_k = ["-k", args.k]
            pytest_m = ["-m", "not java"]
            if args.run_all:
                logger.warning("the argument '-k' overwrites '--run-all'.")
        elif args.run_all:
            pytest_k = []
            pytest_m = []
        else:
            pytest_k = ["-k", "not sniff_test"]
            pytest_m = ["-m", "not java"]

        exit_code = pytest.main([test_dir, "--color=no"] + pytest_k + pytest_m)
        sys.exit(exit_code)
