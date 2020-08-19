"""
The `print-results` subcommand allows you to print results from multiple
allennlp serialization directories to the console in a helpful csv format.
"""

import argparse
import json
import logging
import os

from overrides import overrides

from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)


@Subcommand.register("print-results")
class PrintResults(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:

        description = """Print results from allennlp training runs in a helpful CSV format."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Print results from allennlp serialization directories to the console.",
        )
        subparser.add_argument(
            "path",
            type=str,
            help="Path to recursively search for allennlp serialization directories.",
        )

        subparser.add_argument(
            "-k",
            "--keys",
            type=str,
            nargs="+",
            help="Keys to print from metrics.json."
            'Keys not present in all metrics.json will result in "N/A"',
            default=None,
            required=False,
        )
        subparser.add_argument(
            "-m",
            "--metrics-filename",
            type=str,
            help="Name of the metrics file to inspect.",
            default="metrics.json",
            required=False,
        )

        subparser.set_defaults(func=print_results_from_args)
        return subparser


def print_results_from_args(args: argparse.Namespace):
    """
    Prints results from an `argparse.Namespace` object.
    """
    path = args.path
    metrics_name = args.metrics_filename
    keys = args.keys

    results_dict = {}
    for root, _, files in os.walk(path):
        if metrics_name in files:
            full_name = os.path.join(root, metrics_name)
            with open(full_name) as file_:
                metrics = json.load(file_)
            results_dict[full_name] = metrics

    sorted_keys = sorted(list(results_dict.keys()))
    print(f"model_run, {', '.join(keys)}")
    for name in sorted_keys:
        results = results_dict[name]
        keys_to_print = (str(results.get(key, "N/A")) for key in keys)
        print(f"{name}, {', '.join(keys_to_print)}")
