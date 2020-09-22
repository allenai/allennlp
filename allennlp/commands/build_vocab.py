"""
Subcommand for building a vocabulary from a training config.
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params
from allennlp.training.util import make_vocab_from_params


logger = logging.getLogger(__name__)


@Subcommand.register("build-vocab")
class BuildVocab(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Build a vocabulary from an experiment config file."""
        subparser = parser.add_parser(self.name, description=description, help=description)

        subparser.add_argument("param_path", type=str, help="path to an experiment config file")

        subparser.add_argument(
            "output_path", type=str, help="path to save the vocab tar.gz file to"
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="force write if the output_path already exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.set_defaults(func=build_vocab_from_args)

        return subparser


def build_vocab_from_args(args: argparse.Namespace):
    if not args.output_path.endswith(".tar.gz"):
        raise ValueError("param 'output_path' should end with '.tar.gz'")

    if os.path.exists(args.output_path) and not args.force:
        raise RuntimeError(f"{args.output_path} already exists. Use --force to overwrite.")

    output_directory = os.path.dirname(args.output_path)
    os.makedirs(output_directory, exist_ok=True)

    params = Params.from_file(args.param_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Serializes the vocab to 'tempdir/vocabulary'.
        make_vocab_from_params(params, temp_dir)

        # We create a temp file to archive the vocab to in case something goes wrong
        # while creating the archive so we're not left with a corrupted file.
        temp_archive = tempfile.NamedTemporaryFile(
            suffix=".tar.gz", dir=output_directory, delete=False
        )

        try:
            logger.info("Archiving vocabulary to %s", args.output_path)

            with tarfile.open(temp_archive.name, "w:gz") as archive:
                vocab_dir = os.path.join(temp_dir, "vocabulary")
                for fname in os.listdir(vocab_dir):
                    if fname.endswith(".lock"):
                        continue
                    archive.add(os.path.join(vocab_dir, fname), arcname=fname)

            # Archive successful, now replace the temp file with the target output path.
            os.replace(temp_archive.name, args.output_path)
        finally:
            # Clean up.
            if os.path.exists(temp_archive.name):
                os.remove(temp_archive.name)

    print(f"Success! Vocab saved to {args.output_path}")
    print('You can now set the "vocabulary" entry of your training config to:')
    print(json.dumps({"type": "from_files", "directory": os.path.abspath(args.output_path)}))
