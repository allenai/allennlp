#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.commands.train import Train
from allennlp.common.params import Params

#TODO(michaels): add CLI support for mounting datasets.

def main(param_file, description):
    ecr_repository = "896129387501.dkr.ecr.us-west-2.amazonaws.com"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"{ecr_repository}/allennlp/allennlp:{commit}"
    overrides = ""

    # Read params and set environment
    params = Params.from_file(param_file, overrides)
    flat_params = params.as_flat_dict()
    env = []
    for k, v in flat_params.items():
        k = str(k).replace('.', '_')
        env.append(f"--env={k}={v}")

    # Get temporary ecr login. For this command to work, you need the python awscli
    # package with a version more recent than 1.11.91.
    print("Logging into ECR")
    subprocess.run('eval $(aws --region=us-west-2 ecr get-login --no-include-email)', shell=True, check=True)

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
            f"{config_dataset_id}:/config.json",
            f'--desc={description}',
            '--gpu-count=1',
            '--detach'] + [image] + allennlp_command
    print(' '.join(command))
    subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO(michaels): rename to param-file
    parser.add_argument('params', type=str, help='The model configuration file.')
    parser.add_argument('description', type=str, help='A description for the experiment.')

    args = parser.parse_args()
    main(args.params, args.description)
