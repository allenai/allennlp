"""
Subcommand for running Tango experiments
"""

import argparse
import logging
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Any, List, Optional

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.steps.step import step_graph_from_params
from allennlp.steps.step_cache import DirectoryStepCache

logger = logging.getLogger(__name__)


@Subcommand.register("tango")
class Tango(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Run a tango experiment file."""
        subparser = parser.add_parser(self.name, description=description, help=description)

        subparser.add_argument("config_path", type=str, help="path to a Tango experiment file")

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the results of the steps",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"vocabulary.min_count.labels\": 10}'. Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only show what would run. Don't run anything.",
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=run_tango_from_args)

        return subparser


def run_tango_from_args(args: argparse.Namespace):
    run_tango_from_file(
        tango_filename=args.config_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
    )


def run_tango_from_file(
    tango_filename: Union[str, PathLike],
    serialization_dir: Union[str, PathLike],
    overrides: Union[str, Dict[str, Any]] = "",
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
):
    params = Params.from_file(tango_filename, overrides)
    return run_tango(
        params=params,
        serialization_dir=serialization_dir,
        include_package=include_package,
        dry_run=dry_run,
        file_friendly_logging=file_friendly_logging,
    )


def run_tango(
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: Optional[List[str]] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
):
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    if include_package is not None:
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

    common_util.prepare_environment(params)

    step_graph = step_graph_from_params(params.pop("steps"))

    serialization_dir = Path(serialization_dir)
    serialization_dir.mkdir(parents=True, exist_ok=True)
    step_cache = DirectoryStepCache(serialization_dir / "step_cache")

    # produce results
    for name, step in step_graph.items():
        if step.produce_results:
            _ = step.result(step_cache)

    # symlink everything that has been computed
    for name, step in step_graph.items():
        if step in step_cache:
            (serialization_dir / name).symlink_to(
                step_cache.path_for_step(step), target_is_directory=True
            )
