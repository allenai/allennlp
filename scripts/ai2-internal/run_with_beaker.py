#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import random
import subprocess
import sys
from typing import List

# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.commands.train import Train
from allennlp.common.params import Params

def main(param_file: str, extra_beaker_commands: List[str]):
    ecr_repository = "896129387501.dkr.ecr.us-west-2.amazonaws.com"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"{ecr_repository}/allennlp/allennlp:{commit}"
    overrides = ""

    # Reads params and sets environment.
    params = Params.from_file(param_file, overrides)
    flat_params = params.as_flat_dict()
    env = []
    for k, v in flat_params.items():
        k = str(k).replace('.', '_')
        env.append(f"--env={k}={v}")

    # If the git repository is dirty, add a random hash.
    result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        image += "-" + dirty_hash

    # Get temporary ecr login. For this command to work, you need the python awscli
    # package with a version more recent than 1.11.91.
    print("Generating ECR Login Command")
    login_command = subprocess.check_output('aws --region=us-west-2 ecr get-login --no-include-email', shell=True)

    print("Logging into ECR")
    subprocess.run(login_command, shell=True, check=True)

    print(f"Building the Docker image ({image})")
    subprocess.run(f'docker build -t {image} .', shell=True, check=True)

    print(f"Pushing the Docker image ({image})")
    subprocess.run(f'docker push {image}', shell=True, check=True)

    config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {param_file}', shell=True, universal_newlines=True).strip()
    filename = os.path.basename(param_file)

    allennlp_command = [
            "python",
            "-m",
            "allennlp.run",
            "train",
            "/config.json",
            "-s",
            "/output",
            "--file-friendly-logging"
        ]

    # TODO(michaels): add back in the env list.
    # Presently this makes the Beaker UI unusably cluttered.
    command = [
            '/usr/local/bin/beaker',
            'experiment',
            'run',
            '--result-path',
            '/output',
            "--source",
            f"{config_dataset_id}:/config.json"] + env + extra_beaker_commands + [image] + allennlp_command
    print(' '.join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str, help='The model configuration file.')
    parser.add_argument('--name', type=str, help='A name for the experiment.')
    parser.add_argument('--desc', type=str, help='A description for the experiment.')
    parser.add_argument('--debug', action='store_true', help='Print verbose stack traces on error.')
    parser.add_argument('--env', action='append', help='Set environment variables (e.g. NAME=value or NAME)')
    parser.add_argument('--mount', action='append', help='Bind a host directory (e.g. /host/path:/target/path)')
    parser.add_argument('--source', action='append', help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--cpu', help='CPUs to reserve for this experiment (e.g., 0.5)')
    parser.add_argument('--gpu_count', help='GPUs to use for this experiment (e.g., 1 (default))')
    parser.add_argument('--memory', help='Memory to reserve for this experiment (e.g., 1GB)')

    args = parser.parse_args()

    extra_beaker_commands = []
    if args.desc:
        extra_beaker_commands.append(f'--desc={args.desc}')
    if args.name:
        # Remove spaces from the name, because Beaker doesn't allow them.
        extra_beaker_commands.append(f'--name={args.name.replace(" ", "-")}')
    if args.debug:
        extra_beaker_commands.append("--debug")
    if args.env:
        extra_beaker_commands.extend([f"--env={env}" for env in args.env])
    if args.mount:
        extra_beaker_commands.extend([f"--mount={mount}" for mount in args.mount])
    if args.source:
        extra_beaker_commands.extend([f"--source={source}" for source in args.source])
    if args.cpu:
        extra_beaker_commands.append(f"--cpu={args.cpu}")
    if args.gpu_count:
        extra_beaker_commands.append(f"--gpu-count={args.gpu_count}")
    else:
        extra_beaker_commands.append(f"--gpu-count=1")
    if args.memory:
        extra_beaker_commands.append(f"--memory={args.memory}")

    main(args.param_file, extra_beaker_commands)
