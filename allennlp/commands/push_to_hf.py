"""
The `push_to_hf` subcommand can be used to push a trained model to the
Hugging Face Hub ([hf.co](https://hf.co/)).
"""

import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.push_to_hf import push_to_hf

logger = logging.getLogger(__name__)


@Subcommand.register("push_to_hf")
class PushToHf(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Push a model to the Hugging Face Hub

        Pushing your models to the Hugging Face Hub ([hf.co](https://hf.co/)) 
        allows you to share your models with others. On top of that, you can try
        the models directly in the browser with the available widgets.

        Before running this command, login to Hugging Face with
            ```
            pip install huggingface_hub
            huggingface-cli login
            ```
        """
        subparser = parser.add_parser(self.name, description=description, help=description)
        subparser.set_defaults(func=push)

        subparser.add_argument(
            "-a",
            "--archive_path",
            required=True,
            type=str,
            help="full path to the zipped model or to a directory with the serialized model.",
        )

        subparser.add_argument(
            "-n",
            "--repo_name",
            required=True,
            type=str,
            default="Name of the repository",
            help="Name of the repository",
        )

        subparser.add_argument(
            "-o",
            "--organization",
            required=False,
            type=str,
            help="name of organization to which the model should be uploaded",
        )

        subparser.add_argument(
            "-c",
            "--commit_message",
            required=False,
            type=str,
            default="Update repository",
            help="Commit message to use for the push",
        )

        subparser.add_argument(
            "-l",
            "--local_repo_path",
            required=False,
            type=str,
            default="hub",
            help="local path for creating repo",
        )

        return subparser


def push(args: argparse.Namespace):
    push_to_hf(
        args.archive_path,
        args.repo_name,
        args.organization,
        args.commit_message,
        args.local_repo_path,
    )
