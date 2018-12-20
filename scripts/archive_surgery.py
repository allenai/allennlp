#! /usr/bin/env python
"""
Helper script for modifying config.json files that are locked inside
model.tar.gz archives. This is useful if you need to rename things or
add or remove values, usually because of changes to the library.
"""
# pylint: disable=invalid-name,redefined-outer-name
import argparse
import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import tarfile

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.models.archival import CONFIG_NAME, _FTA_NAME, _WEIGHTS_NAME, archive_model

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# The actual surgery
def move(args: argparse.Namespace, config: Params) -> Params:
    """
    Move the value at `old_key` (which could be a compound key
    like "trainer.num_epochs") to `new_key`. Raises an error
    if the old key doesn't exist or if the new key already does.
    """
    old_key = args.old_key
    new_key = args.new_key

    top_dict = config.as_dict()
    curr_dict = top_dict

    old_kps = old_key.split(".")
    for kp in old_kps[:-1]:
        if kp not in curr_dict:
            raise ValueError(f"old-key {old_key} does not exist in config file!")
        curr_dict = curr_dict[kp]

    if old_kps[-1] not in curr_dict:
        raise ValueError(f"old-key {old_key} does not exist in config file!")
    value = curr_dict.pop(old_kps[-1])

    curr_dict = top_dict
    new_kps = new_key.split(".")
    for kp in new_kps[:-1]:
        if kp not in curr_dict:
            curr_dict[kp] = {}
        curr_dict = curr_dict[kp]

    if new_kps[-1] in curr_dict:
        raise ValueError(f"new-key {new_key} already exists in config file!")
    curr_dict[new_kps[-1]] = value

    return Params(top_dict)

def set_value(args: argparse.Namespace, config: Params) -> Params:
    """
    Set the value at the specified (compound) key.

    If we're doing "add", creates intermediate dicts if they don't exist,
    and raises an error if the key already exists.

    If we're doing "replace", raises an error if any of the keys don't exist.
    """
    key = args.key
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    command = args.command

    top_dict = config.as_dict()
    curr_dict = top_dict

    kps = key.split(".")
    for kp in kps[:-1]:
        if kp not in curr_dict:
            if command == "add":
                curr_dict[kp] = {}
            else:
                raise ValueError(f"key does not exist: {key}")
        curr_dict = curr_dict[kp]

    if command == "add" and kps[-1] in curr_dict:
        raise ValueError(f"key {key} already exists in config file!")
    curr_dict[kps[-1]] = value

    return Params(top_dict)

def remove(args: argparse.Namespace, config: Params) -> Params:
    """
    Remove the subconfig or item at the specified (compound) key.
    Raises an error if the key doesn't exist.
    """
    key = args.key

    top_dict = config.as_dict()
    curr_dict = top_dict

    kps = key.split(".")
    for kp in kps[:-1]:
        if kp not in curr_dict:
            raise ValueError(f"key {key} does not exist in config file")
        curr_dict = curr_dict[kp]

    if kps[-1] not in curr_dict:
        raise ValueError(f"key {key} does not exist in config file")
    del curr_dict[kps[-1]]

    return Params(top_dict)

def inspect(args: argparse.Namespace, config: Params) -> Params:
    """
    Just print out all the flattened key-values in the config
    """
    flat = config.as_flat_dict()
    for key, value in flat.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Perform surgery on a model.tar.gz archive")

    subparsers = parser.add_subparsers(title='Commands', metavar='', help='commands', dest='command')

    move_parser = subparsers.add_parser('move', help="move a value from one key to another")
    add_parser = subparsers.add_parser('add', help="add a new key-value pair")
    remove_parser = subparsers.add_parser('remove', help="remove a key-value pair")
    replace_parser = subparsers.add_parser('replace', help="replace a value")
    inspect_parser = subparsers.add_parser('inspect', help="inspect a config file")

    move_parser.add_argument("--old-key", required=True)
    move_parser.add_argument("--new-key", required=True)
    move_parser.add_argument("--input-file", required=True)
    move_parser.add_argument("--output-file", required=True)
    move_parser.set_defaults(func=move)

    add_parser.add_argument("--key", required=True)
    add_parser.add_argument("--value", required=True)
    add_parser.add_argument("--input-file", required=True)
    add_parser.add_argument("--output-file", required=True)
    add_parser.set_defaults(func=set_value)

    replace_parser.add_argument("--key", required=True)
    replace_parser.add_argument("--value", required=True)
    replace_parser.add_argument("--input-file", required=True)
    replace_parser.add_argument("--output-file", required=True)
    replace_parser.set_defaults(func=set_value)

    remove_parser.add_argument("--key", required=True)
    remove_parser.add_argument("--input-file", required=True)
    remove_parser.add_argument("--output-file", required=True)
    remove_parser.set_defaults(func=remove)

    inspect_parser.add_argument("--input-file", required=True)
    inspect_parser.set_defaults(func=inspect)

    args = parser.parse_args()
    command = args.command

    if command != "inspect" and os.path.exists(args.output_file):
        raise ValueError("output file already exists")

    # redirect to the cache, if necessary
    archive_file = cached_path(args.input_file)

    if not os.path.exists(archive_file):
        raise ValueError("input file doesn't exist")

    # Extract archive to temp dir
    tempdir = tempfile.mkdtemp()
    with tarfile.open(archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    atexit.register(lambda: shutil.rmtree(tempdir))

    # Check for supplemental files in archive
    fta_filename = os.path.join(tempdir, _FTA_NAME)
    if os.path.exists(fta_filename):
        with open(fta_filename, 'r') as fta_file:
            files_to_archive = json.loads(fta_file.read())
    else:
        files_to_archive = {}

    # Replace with paths in the tempdir
    files_to_archive = {key: os.path.join(tempdir, f"fta/{key}")
                        for key in files_to_archive}
    overrides = json.dumps(files_to_archive)

    # Load config
    config_path = os.path.join(tempdir, CONFIG_NAME)
    original_config = Params.from_file(config_path, overrides)

    # Do surgery
    new_config: Params = args.func(args, original_config)

    # If we're just inspecting, we can quit now
    if command == "inspect":
        sys.exit(0)

    # Otherwise let's print out the new config
    inspect(args, new_config)

    # Write out config
    new_config.to_file(config_path)

    #### Generate new archive

    # create archive
    archive_model(tempdir, _WEIGHTS_NAME, files_to_archive=files_to_archive)

    # and move it to final destination
    shutil.move(os.path.join(tempdir, 'model.tar.gz'), args.output_file)

if __name__ == "__main__":
    main()
