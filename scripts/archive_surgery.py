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
# pylint: disable=invalid-name,redefined-outer-name
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
    parser = argparse.ArgumentParser(description="Perform surgery on a model.tar.gz archive")

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--editor")

    args = parser.parse_args()

    editor = args.editor or os.environ.get("EDITOR")
    if editor is None:
        raise RuntimeError("please specify an editor or set the $EDITOR environment variable")

    if os.path.exists(args.output_file):
        raise ValueError("output file already exists")

    archive_file = cached_path(args.input_file)
    if not os.path.exists(archive_file):
        raise ValueError("input file doesn't exist")

    # Extract archive to temp dir
    tempdir = tempfile.mkdtemp()
    with tarfile.open(archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    atexit.register(lambda: shutil.rmtree(tempdir))

    config_path = os.path.join(tempdir, CONFIG_NAME)
    subprocess.run([editor, config_path])

    with tarfile.open(args.output_file, "w:gz") as tar:
        tar.add(tempdir, arcname=os.path.sep)


if __name__ == "__main__":
    main()
