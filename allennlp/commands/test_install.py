"""
The ``test-install`` subcommand verifies
an installation by running the unit tests.

.. code-block:: bash

    $ allennlp test-install --help
    usage: allennlp test-install [-h] [--run-all]
                                 [--include-package INCLUDE_PACKAGE]

    Test that installation works by running the unit tests.

    optional arguments:
      -h, --help            show this help message and exit
      --run-all             By default, we skip tests that are slow or download
                            large files. This flag will run all tests.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

import argparse
import logging
import os
import pathlib

import pytest

import allennlp
from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TestInstall(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Test that installation works by running the unit tests.'''
        subparser = parser.add_parser(
                name, description=description, help='Run the unit tests.')

        subparser.add_argument('--run-all', action="store_true",
                               help="By default, we skip tests that are slow "
                               "or download large files. This flag will run all tests.")
        subparser.add_argument('-k', type=str, default=None,
                               help="Limit tests by setting pytest -k argument")

        subparser.set_defaults(func=_run_test)

        return subparser


def _get_module_root():
    return pathlib.Path(allennlp.__file__).parent


def _run_test(args: argparse.Namespace):
    initial_working_dir = os.getcwd()
    module_parent = _get_module_root().parent
    logger.info("Changing directory to %s", module_parent)
    os.chdir(module_parent)
    test_dir = os.path.join(module_parent, "allennlp")
    logger.info("Running tests at %s", test_dir)

    if args.k:
        pytest_k = ['-k', args.k]
        pytest_m = ['-m', 'not java']
        if args.run_all:
            logger.warning("the argument '-k' overwrites '--run-all'.")
    elif args.run_all:
        pytest_k = []
        pytest_m = []
    else:
        pytest_k = ['-k', 'not sniff_test']
        pytest_m = ['-m', 'not java']

    exit_code = pytest.main([test_dir, '--color=no'] + pytest_k + pytest_m)

    # Change back to original working directory after running tests
    os.chdir(initial_working_dir)
    exit(exit_code)
