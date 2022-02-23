"""
The `push-to-hf` subcommand can be used to push a trained model to the
Hugging Face Hub ([hf.co](https://hf.co/)).
"""

import argparse
import logging


from allennlp.commands.subcommand import Subcommand
from allennlp.common.push_to_hf import push_to_hf

logger = logging.getLogger(__name__)


@Subcommand.register("push-to-hf")
class PushToHf(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Push a model to the Hugging Face Hub.

        Pushing your models to the Hugging Face Hub ([hf.co](https://hf.co/))
        allows you to share your models with others. On top of that, you can try
        the models directly in the browser with the available widgets.

        Before running this command, login to Hugging Face with `huggingface-cli login`.

        You can specify either a `serialization_dir` or an `archive_path`, but using the
        first option is recommended since the `serialization_dir` contains more useful
        information such as metrics and TensorBoard traces.
        """
        subparser = parser.add_parser(self.name, description=description, help=description)
        subparser.set_defaults(func=push)

        subparser.add_argument(
            "-n",
            "--repo-name",
            required=True,
            type=str,
            default="Name of the repository",
            help="Name of the repository",
        )

        model_dir_group = subparser.add_mutually_exclusive_group(required=True)
        model_dir_group.add_argument(
            "-s",
            "--serialization-dir",
            type=str,
            help="directory in which to save the model and its logs are saved",
        )

        model_dir_group.add_argument(
            "-a",
            "--archive-path",
            type=str,
            help="full path to the zipped model, using serialization_dir instead is recommended",
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
            "--commit-message",
            required=False,
            type=str,
            default="Update repository",
            help="Commit message to use for the push",
        )

        subparser.add_argument(
            "-l",
            "--local-repo-path",
            required=False,
            type=str,
            default="hub",
            help="local path for creating repo",
        )

        return subparser


def push(args: argparse.Namespace):
    push_to_hf(
        args.repo_name,
        serialization_dir=args.serialization_dir,
        archive_path=args.archive_path,
        organization=args.organization,
        commit_message=args.commit_message,
        local_repo_path=args.local_repo_path,
    )
