"""
CLI to the the caching mechanism in `common.file_utils`.
"""

import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import (
    cached_path,
    CACHE_DIRECTORY,
    inspect_cache,
    remove_cache_entries,
)


logger = logging.getLogger(__name__)


@Subcommand.register("cached-path")
class CachedPath(Subcommand):
    requires_plugins: bool = False

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
            help="""The URLs or paths to the resources.
            If using the --inspect or --remove flag, this can also contain glob patterns.""",
            nargs="*",
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
        subparser.add_argument(
            "--inspect",
            action="store_true",
            help="""Print some useful information about the cache.""",
        )
        subparser.add_argument(
            "--remove",
            action="store_true",
            help="""Remove any cache entries matching the given resource patterns.""",
        )
        return subparser


def _cached_path(args: argparse.Namespace):
    logger.info("Cache directory: %s", args.cache_dir)
    if args.inspect:
        if args.extract_archive or args.force_extract or args.remove:
            raise RuntimeError(
                "cached-path cannot accept --extract-archive, --force-extract, or --remove "
                "options when --inspect flag is used."
            )
        inspect_cache(patterns=args.resources, cache_dir=args.cache_dir)
    elif args.remove:
        from allennlp.common.util import format_size

        if args.extract_archive or args.force_extract or args.inspect:
            raise RuntimeError(
                "cached-path cannot accept --extract-archive, --force-extract, or --inspect "
                "options when --remove flag is used."
            )
        if not args.resources:
            raise RuntimeError(
                "Missing positional argument(s) 'resources'. 'resources' is required when using "
                "the --remove option. If you really want to remove everything, pass '*' for 'resources'."
            )
        reclaimed_space = remove_cache_entries(args.resources, cache_dir=args.cache_dir)
        print(f"Reclaimed {format_size(reclaimed_space)} of space")
    else:
        for resource in args.resources:
            print(
                cached_path(
                    resource,
                    cache_dir=args.cache_dir,
                    extract_archive=args.extract_archive,
                    force_extract=args.force_extract,
                )
            )
