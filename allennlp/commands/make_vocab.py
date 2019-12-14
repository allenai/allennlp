import argparse
import logging
import warnings

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.dry_run import dry_run_from_args

logger = logging.getLogger(__name__)


class MakeVocab(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """make-vocab is deprecated. Use dry-run instead (with the same arguments!)"""
        subparser = parser.add_parser(name, description=description, help=description)
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
            help="directory in which to save the vocabulary directory",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        def warn_and_run_dry_run(args: argparse.Namespace):
            warnings.warn(
                "make-vocab is deprecated. Use dry-run instead (with the same arguments!)",
                FutureWarning,
            )

            return dry_run_from_args(args)

        subparser.set_defaults(func=warn_and_run_dry_run)

        return subparser
