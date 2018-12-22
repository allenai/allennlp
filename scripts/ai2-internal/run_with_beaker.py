#! /usr/bin/env python

# Script to launch AllenNLP Beaker jobs.

import argparse
import os
import json
import random
import tempfile
import subprocess
import sys

# This has to happen before we import spacy (even indirectly), because for some crazy reason spacy
# thought it was a good idea to set the random seed on import...
random_int = random.randint(0, 2**32)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.params import Params


def main(param_file: str, args: argparse.Namespace):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    image = f"allennlp/allennlp:{commit}"
    overrides = ""

    # Reads params and sets environment.
    ext_vars = {}

    for var in args.env:
        key, value = var.split("=")
        ext_vars[key] = value

    params = Params.from_file(param_file, overrides, ext_vars)
    flat_params = params.as_flat_dict()
    env = {}
    for k, v in flat_params.items():
        k = str(k).replace('.', '_')
        env[k] = str(v)

    # If the git repository is dirty, add a random hash.
    result = subprocess.run('git diff-index --quiet HEAD --', shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        image += "-" + dirty_hash

    if args.blueprint:
        blueprint = args.blueprint
        print(f"Using the specified blueprint: {blueprint}")
    else:
        print(f"Building the Docker image ({image})...")
        subprocess.run(f'docker build -t {image} .', shell=True, check=True)

        print(f"Create a Beaker blueprint...")
        blueprint = subprocess.check_output(f'beaker blueprint create --quiet {image}', shell=True, universal_newlines=True).strip()
        print(f"  Blueprint created: {blueprint}")

    config_dataset_id = subprocess.check_output(f'beaker dataset create --quiet {param_file}', shell=True, universal_newlines=True).strip()

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

    dataset_mounts = []
    for source in args.source + [f"{config_dataset_id}:/config.json"]:
        datasetId, containerPath = source.split(":")
        dataset_mounts.append({
            "datasetId": datasetId,
            "containerPath": containerPath
        })

    for var in args.env:
        key, value = var.split("=")
        env[key] = value

    requirements = {}
    if args.cpu:
        requirements["cpu"] = float(args.cpu)
    if args.memory:
        requirements["memory"] = args.memory
    if args.gpu_count:
        requirements["gpuCount"] = int(args.gpu_count)
    config_spec = {
        "description": args.desc,
        "blueprint": blueprint,
        "resultPath": "/output",
        "args": allennlp_command,
        "datasetMounts": dataset_mounts,
        "requirements": requirements,
        "env": env
    }
    config_task = {"spec": config_spec, "name": "training"}

    config = {
        "tasks": [config_task]
    }

    output_path = args.spec_output_path if args.spec_output_path else tempfile.mkstemp(".yaml",
            "beaker-config-")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--file", output_path]
    if args.name:
        experiment_command.append("--name")
        experiment_command.append(args.name.replace(" ", "-"))

    if args.dry_run:
        print(f"This is a dry run (--dry-run).  Launch your job with the following command:")
        print(f"    " + " ".join(experiment_command))
    else:
        print(f"Running the experiment:")
        print(f"    " + " ".join(experiment_command))
        subprocess.run(experiment_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('param_file', type=str, help='The model configuration file.')
    parser.add_argument('--name', type=str, help='A name for the experiment.')
    parser.add_argument('--spec_output_path', type=str, help='The destination to write the experiment spec.')
    parser.add_argument('--dry-run', action='store_true', help='If specified, an experiment will not be created.')
    parser.add_argument('--blueprint', type=str, help='The Blueprint to use (if unspecified one will be built)')
    parser.add_argument('--desc', type=str, help='A description for the experiment.')
    parser.add_argument('--env', action='append', default=[], help='Set environment variables (e.g. NAME=value or NAME)')
    parser.add_argument('--source', action='append', default=[], help='Bind a remote data source (e.g. source-id:/target/path)')
    parser.add_argument('--cpu', help='CPUs to reserve for this experiment (e.g., 0.5)')
    parser.add_argument('--gpu-count', default=1, help='GPUs to use for this experiment (e.g., 1 (default))')
    parser.add_argument('--memory', help='Memory to reserve for this experiment (e.g., 1GB)')

    args = parser.parse_args()

    main(args.param_file, args)
