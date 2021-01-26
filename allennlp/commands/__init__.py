import argparse
import logging
import sys
from typing import Any, Optional, Tuple, Set

from overrides import overrides

from allennlp import __version__
from allennlp.commands.build_vocab import BuildVocab
from allennlp.commands.cached_path import CachedPath
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.find_learning_rate import FindLearningRate
from allennlp.commands.predict import Predict
from allennlp.commands.print_results import PrintResults
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.train import Train
from allennlp.commands.count_instances import CountInstances
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules

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
            description = kwargs.get("help", "")
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def parse_args(prog: Optional[str] = None) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Creates the argument parser for the main program and uses it to parse the args.
    """
    parser = ArgumentParserWithDefaults(description="Run AllenNLP", prog=prog)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands: Set[str] = set()

    def add_subcommands():
        for subcommand_name in sorted(Subcommand.list_available()):
            if subcommand_name in subcommands:
                continue
            subcommands.add(subcommand_name)
            subcommand_class = Subcommand.by_name(subcommand_name)
            subcommand = subcommand_class()
            subparser = subcommand.add_subparser(subparsers)
            if subcommand_class.requires_plugins:
                subparser.add_argument(
                    "--include-package",
                    type=str,
                    action="append",
                    default=[],
                    help="additional packages to include",
                )

    # Add all default registered subcommands first.
    add_subcommands()

    # If we need to print the usage/help, or the subcommand is unknown,
    # we'll call `import_plugins()` to register any plugin subcommands first.
    argv = sys.argv[1:]
    plugins_imported: bool = False
    if not argv or argv == ["--help"] or argv[0] not in subcommands:
        import_plugins()
        plugins_imported = True
        # Add subcommands again in case one of the plugins has a registered subcommand.
        add_subcommands()

    # Now we can parse the arguments.
    args = parser.parse_args()

    if not plugins_imported and Subcommand.by_name(argv[0]).requires_plugins:  # type: ignore
        import_plugins()

    return parser, args


def main(prog: Optional[str] = None) -> None:
    """
    The [`run`](./train.md#run) command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own `Model` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see [`plugins`](./plugins.md)).
    """
    parser, args = parse_args(prog)

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, "include_package", []):
            import_module_and_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
