"""
The `checklist` subcommand allows you to sanity check your
model's predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""

from typing import Optional, Dict, Any
import argparse
import sys
import json

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite


@Subcommand.register("checklist")
class CheckList(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:

        description = """Run the specified model through a checklist suite."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Run a trained model through a checklist suite.",
        )

        subparser.add_argument(
            "archive_file", type=str, help="the archived model to make predictions with"
        )
        subparser.add_argument("task_suite", type=str, help="the suite name or path")

        subparser.add_argument(
            "--task-suite-args",
            type=str,
            default="",
            help=(
                "an optional JSON structure used to provide additional parameters to the task suite"
            ),
        )

        subparser.add_argument(
            "--print-summary-args",
            type=str,
            default="",
            help=(
                "an optional JSON structure used to provide additional "
                "parameters for printing test summary"
            ),
        )

        subparser.add_argument("--output-file", type=str, help="path to output file")

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--predictor", type=str, help="optionally specify a specific predictor to use"
        )

        subparser.add_argument(
            "--predictor-args",
            type=str,
            default="",
            help=(
                "an optional JSON structure used to provide additional parameters to the predictor"
            ),
        )

        subparser.set_defaults(func=_run_suite)

        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
        args.archive_file,
        cuda_device=args.cuda_device,
    )

    predictor_args = args.predictor_args.strip()
    if len(predictor_args) <= 0:
        predictor_args = {}
    else:
        predictor_args = json.loads(predictor_args)

    return Predictor.from_archive(
        archive,
        args.predictor,
        extra_args=predictor_args,
    )


def _get_task_suite(args: argparse.Namespace) -> TaskSuite:
    if args.task_suite in TaskSuite.list_available():
        suite_name = args.task_suite
        file_path = None
    else:
        suite_name = None
        file_path = args.task_suite

    task_suite_args = args.task_suite_args.strip()
    if len(task_suite_args) <= 0:
        task_suite_args = {}
    else:
        task_suite_args = json.loads(task_suite_args)

    return TaskSuite.constructor(
        name=suite_name,
        suite_file=file_path,
        extra_args=task_suite_args,
    )


class _CheckListManager:
    def __init__(
        self,
        task_suite: TaskSuite,
        predictor: Predictor,
        output_file: Optional[str],
        print_summary_args: Optional[Dict[str, Any]],
    ) -> None:
        self._task_suite = task_suite
        self._predictor = predictor
        self._output_file = None if output_file is None else open(output_file, "w")
        self._print_summary_args = print_summary_args or {}

    def run(self) -> None:
        self._task_suite.run(self._predictor)
        output_file = self._output_file or sys.stdout
        self._task_suite.summary(file=output_file, **self._print_summary_args)

        if self._output_file is not None:
            self._output_file.close()


def _run_suite(args: argparse.Namespace) -> None:

    task_suite = _get_task_suite(args)
    predictor = _get_predictor(args)

    print_summary_args = args.print_summary_args.strip()
    if len(print_summary_args) <= 0:
        print_summary_args = {}
    else:
        print_summary_args = json.loads(print_summary_args)

    manager = _CheckListManager(
        task_suite,
        predictor,
        args.output_file,
        print_summary_args,
    )
    manager.run()
