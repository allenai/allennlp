#! /usr/bin/env python
"""
Helper script for modifying config.json files that are locked inside
model.tar.gz archives. This is useful if you need to rename things or
add or remove values, usually because of changes to the library.

This script will untar the archive to a temp directory, launch an editor
to modify the config.json, and then re-tar everything to a new archive.
If your $EDITOR environment variable is not set, you'll have to explicitly
specify which editor to use.
"""

import argparse
import atexit
import logging
import os
import shutil
import subprocess
import tempfile
import tarfile

from allennlp.common.file_utils import cached_path
from allennlp.models.archival import CONFIG_NAME

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="Perform surgery on a model.tar.gz archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-file", required=True, help="path to input file")
    parser.add_argument(
        "--editor",
        default=os.environ.get("EDITOR"),
        help="editor to launch, whose default value is `$EDITOR` the environment variable",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-file", help="path to output file")
    output.add_argument(
        "--inplace",
        action="store_true",
        help="overwrite the input file with the modified configuration",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite the output file if it exists"
    )

    args = parser.parse_args()

    if args.editor is None:
        raise RuntimeError("please specify an editor or set the $EDITOR environment variable")

    if not args.inplace and os.path.exists(args.output_file) and not args.force:
        raise ValueError("output file already exists, use --force to override")

    archive_file = cached_path(args.input_file)
    if not os.path.exists(archive_file):
        raise ValueError("input file doesn't exist")
    if args.inplace:
        output_file = archive_file
    else:
        output_file = args.output_file

    # Extract archive to temp dir
    tempdir = tempfile.mkdtemp()
    with tarfile.open(archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    atexit.register(lambda: shutil.rmtree(tempdir))

    config_path = os.path.join(tempdir, CONFIG_NAME)
    subprocess.run([args.editor, config_path], check=False)

    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(tempdir, arcname=os.path.sep)


if __name__ == "__main__":
    main()
