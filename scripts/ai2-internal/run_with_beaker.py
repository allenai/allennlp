#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import subprocess

#TODO(michaels): add CLI support for mounting datasets.

def main(param_file, description):
    result_arg="--result-path /output"
    gpu_arg="--gpu-count=1"
    detach_arg="--detach"

    ecr_repository="896129387501.dkr.ecr.us-west-2.amazonaws.com"

    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image=f"{ecr_repository}/allennlp/allennlp:{commit}"

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

    allennlp_command="allennlp/run train /config.json -s /output --file-friendly-logging"

    command = f'beaker experiment run {result_arg} {description} {gpu_arg} {detach_arg} {image} {allennlp_command}'
    print(command)
    subprocess.run('command', shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO(michaels): rename to param-file
    parser.add_argument('params', type=str, help='The model configuration file.')
    parser.add_argument('description', type=str, help='A description for the experiment.')

    args = parser.parse_args()
    main(args.params, args.description)
