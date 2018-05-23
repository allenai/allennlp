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

import pytest

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

        subparser.set_defaults(func=_run_test)

        return subparser


def _get_project_root():
    return os.path.abspath(
            os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    os.pardir, os.pardir))


def _run_test(args: argparse.Namespace):
    initial_working_dir = os.getcwd()
    project_root = _get_project_root()
    logger.info("Changing directory to %s", project_root)
    os.chdir(project_root)
    test_dir = os.path.join(project_root, "tests")
    logger.info("Running tests at %s", test_dir)
    if args.run_all:
        pytest.main([test_dir])
    else:
        pytest.main([test_dir, '-k', 'not sniff_test and not notebooks_test'])
    # Change back to original working directory after running tests
    os.chdir(initial_working_dir)
