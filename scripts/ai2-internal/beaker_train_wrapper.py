#! /usr/bin/env python

# Script to run a training job for Beaker.  This script helps with the following:
#   1.  Exports training configurations as environment variables
#   2.  Post-processes the output to generate a metrics.json.

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.commands.train import Train
from allennlp.common.params import Params

def main(argv, param_path, overrides):
    # Build environment variable map
    params = Params.from_file(param_path, overrides)
    env = params.as_flat_dict()

    command = ['python', '-m', 'allennlp.run', 'train'] + argv[1:]
    subprocess.run(command, env = {**os.environ, **env}, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Train.add_arguments(parser)

    parsed_args = parser.parse_args()
    main(sys.argv, parsed_args.param_path, parsed_args.overrides)
