"""
The `checklist` subcommand allows you to conduct behavioural
testing for your model's predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""

from typing import Optional, Dict, Any, List
import argparse
import sys
import json

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite


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
            "archive_file", type=str, help="The archived model to make predictions with"
        )

        subparser.add_argument("task", type=str, help="The name of the task suite")

        subparser.add_argument("--checklist-suite", type=str, help="The checklist suite path")

        subparser.add_argument(
            "--capabilities",
            nargs="+",
            default=[],
            help=('An optional list of strings of capabilities. Eg. "[Vocabulary, Robustness]"'),
        )

        subparser.add_argument(
            "--max-examples",
            type=int,
            default=None,
            help="Maximum number of examples to check per test.",
        )

        subparser.add_argument(
            "--task-suite-args",
            type=str,
            default="",
            help=(
                "An optional JSON structure used to provide additional parameters to the task suite"
            ),
        )

        subparser.add_argument(
            "--print-summary-args",
            type=str,
            default="",
            help=(
                "An optional JSON structure used to provide additional "
                "parameters for printing test summary"
            ),
        )

        subparser.add_argument("--output-file", type=str, help="Path to output file")

        subparser.add_argument(
            "--cuda-device", type=int, default=-1, help="ID of GPU to use (if any)"
        )

        subparser.add_argument(
            "--predictor", type=str, help="Optionally specify a specific predictor to use"
        )

        subparser.add_argument(
            "--predictor-args",
            type=str,
            default="",
            help=(
                "An optional JSON structure used to provide additional parameters to the predictor"
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
    available_tasks = TaskSuite.list_available()
    if args.task in available_tasks:
        suite_name = args.task
    else:
        raise ConfigurationError(
            f"'{args.task}' is not a recognized task suite. "
            f"Available tasks are: {available_tasks}."
        )

    file_path = args.checklist_suite

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
        capabilities: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        output_file: Optional[str] = None,
        print_summary_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_suite = task_suite
        self._predictor = predictor
        self._capabilities = capabilities
        self._max_examples = max_examples
        self._output_file = None if output_file is None else open(output_file, "w")
        self._print_summary_args = print_summary_args or {}

        if capabilities:
            self._print_summary_args["capabilities"] = capabilities

    def run(self) -> None:
        self._task_suite.run(
            self._predictor, capabilities=self._capabilities, max_examples=self._max_examples
        )

        # We pass in an IO object.
        output_file = self._output_file or sys.stdout
        self._task_suite.summary(file=output_file, **self._print_summary_args)

        # If `_output_file` was None, there would be nothing to close.
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

    capabilities = args.capabilities
    max_examples = args.max_examples

    manager = _CheckListManager(
        task_suite,
        predictor,
        capabilities,
        max_examples,
        args.output_file,
        print_summary_args,
    )
    manager.run()
