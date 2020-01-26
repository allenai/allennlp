import argparse
import logging
from typing import Any, Iterable

from overrides import overrides

from allennlp import __version__
from allennlp.commands.dry_run import DryRun
from allennlp.commands.elmo import Elmo
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.find_learning_rate import FindLearningRate
from allennlp.commands.fine_tune import FineTune
from allennlp.commands.predict import Predict
from allennlp.commands.print_results import PrintResults
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.train import Train
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_submodules

logger = logging.getLogger(__name__)


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default: Any) -> bool:
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    @overrides
    def add_argument(self, *args, **kwargs):
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get(
            "action"
        ) not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help") or ""
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def create_parser(
    prog: str = None, subcommand_overrides: Iterable[Subcommand] = None
) -> argparse.ArgumentParser:
    """
    Creates the argument parser for the main program.
    """
    subcommand_overrides = subcommand_overrides or []

    parser = ArgumentParserWithDefaults(description="Run AllenNLP", prog=prog)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommand_overrides_names = {subcommand.name for subcommand in subcommand_overrides}

    subcommands = set(subcommand_overrides)

    # The class of each instance in the overrides iterable is gonna be in the subclasses
    # iterable, by definition. So we only add those `Subcommand` subclasses that don't have an
    # instance in the overrides iterable.
    for subcommand_class in Subcommand.__subclasses__():
        subcommand = subcommand_class()
        if subcommand.name not in subcommand_overrides_names:
            subcommands.add(subcommand)

    for subcommand in sorted(list(subcommands), key=lambda s: s.name):
        subparser = subcommand.add_subparser(subparsers)
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )

    return parser


def main(prog: str = None, subcommand_overrides: Iterable[Subcommand] = None) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin.
    """
    subcommand_overrides = subcommand_overrides or []

    import_plugins()

    parser = create_parser(prog, subcommand_overrides)
    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, "include_package", ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
