"""
Subcommand for counting the number of instances from a training config.
"""

import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params


logger = logging.getLogger(__name__)


@Subcommand.register("count-instances")
class CountInstances(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Count the number of training instances in an experiment config file."""
        subparser = parser.add_parser(self.name, description=description, help=description)
        subparser.add_argument("param_path", type=str, help="path to an experiment config file")

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"vocabulary.min_count.labels\": 10}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.set_defaults(func=count_instances_from_args)

        return subparser


def count_instances_from_args(args: argparse.Namespace):
    from allennlp.training.util import data_loaders_from_params

    params = Params.from_file(args.param_path)

    data_loaders = data_loaders_from_params(params, train=True, validation=False, test=False)
    instances = sum(
        1 for data_loader in data_loaders.values() for _ in data_loader.iter_instances()
    )

    print(f"Success! One epoch of training contains {instances} instances.")
