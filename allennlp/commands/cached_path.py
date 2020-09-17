"""
CLI to the the caching mechanism in `common.file_utils`.
"""

import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import cached_path, CACHE_DIRECTORY


logger = logging.getLogger(__name__)


@Subcommand.register("cached-path")
class CachedPath(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Cache remote files to the AllenNLP cache."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help=description,
        )
        subparser.set_defaults(func=_cached_path)
        subparser.add_argument(
            "resources",
            type=str,
            help="""The URLs or paths to the resources.""",
            nargs="+",
        )
        subparser.add_argument(
            "-d",
            "--cache-dir",
            type=str,
            help="""Use a custom cache directory.""",
            default=CACHE_DIRECTORY,
        )
        subparser.add_argument(
            "-x",
            "--extract-archive",
            action="store_true",
            help="""Automatically extract zip or tar.gz archive files.""",
        )
        subparser.add_argument(
            "-f",
            "--force-extract",
            action="store_true",
            help="""Extract archives regardless of whether or not they already exist.""",
        )
        return subparser


def _cached_path(args: argparse.Namespace):
    for resource in args.resources:
        print(
            cached_path(
                resource,
                cache_dir=args.cache_dir,
                extract_archive=args.extract_archive,
                force_extract=args.force_extract,
            )
        )
