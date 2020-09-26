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
random_int = random.randint(0, 2 ** 32)

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))
)

from allennlp.common.params import Params


def main(param_file: str, args: argparse.Namespace):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    docker_image = f"allennlp/allennlp:{commit}"
    overrides = args.overrides

    # Reads params and sets environment.
    ext_vars = {}

    for var in args.env:
        key, value = var.split("=")
        ext_vars[key] = value

    params = Params.from_file(param_file, overrides, ext_vars)

    # Write params as json. Otherwise Jsonnet's import feature breaks.
    params_dir = tempfile.mkdtemp(prefix="config")
    compiled_params_path = os.path.join(params_dir, "config.json")
    params.to_file(compiled_params_path)
    print(f"Compiled jsonnet config written to {compiled_params_path}.")

    flat_params = params.as_flat_dict()
    env = {}
    for k, v in flat_params.items():
        k = str(k).replace(".", "_")
        env[k] = str(v)

    # If the git repository is dirty, add a random hash.
    result = subprocess.run("git diff-index --quiet HEAD --", shell=True)
    if result.returncode != 0:
        dirty_hash = "%x" % random_int
        docker_image += "-" + dirty_hash

    if args.image:
        image = args.image
        print(f"Using the specified image: {image}")
    else:
        print(f"Building the Docker image ({docker_image})...")
        subprocess.run(f"docker build -t {docker_image} .", shell=True, check=True)

        print("Create a Beaker image...")
        image = subprocess.check_output(
            f"beaker image create --quiet {docker_image}", shell=True, universal_newlines=True
        ).strip()
        print(f"  Image created: {docker_image}")

    config_dataset_id = subprocess.check_output(
        f"beaker dataset create --quiet {params_dir}/*", shell=True, universal_newlines=True
    ).strip()

    # Arguments that differ between preemptible and regular machine execution.
    if args.preemptible:
        allennlp_prefix = ["/stage/allennlp/resumable_train.sh", "/output", "/config/config.json"]
    else:
        allennlp_prefix = [
            "python",
            "-m",
            "allennlp.run",
            "train",
            "/config/config.json",
            "-s",
            "/output",
        ]

    # All other arguments
    allennlp_suffix = ["--file-friendly-logging"]
    for package_name in args.include_package:
        allennlp_suffix.append("--include-package")
        allennlp_suffix.append(package_name)

    allennlp_command = allennlp_prefix + allennlp_suffix

    dataset_mounts = []
    for source in args.source + [f"{config_dataset_id}:/config"]:
        datasetId, containerPath = source.split(":")
        dataset_mounts.append({"datasetId": datasetId, "containerPath": containerPath})

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
    if args.preemptible:
        requirements["preemptible"] = True
    config_spec = {
        "description": args.desc,
        "image": image,
        "resultPath": "/output",
        "args": allennlp_command,
        "datasetMounts": dataset_mounts,
        "requirements": requirements,
        "env": env,
    }
    config_task = {"spec": config_spec, "name": "training"}

    config = {"tasks": [config_task]}

    output_path = (
        args.spec_output_path
        if args.spec_output_path
        else tempfile.mkstemp(".yaml", "beaker-config-")[1]
    )
    with open(output_path, "w") as output:
        output.write(json.dumps(config, indent=4))
    print(f"Beaker spec written to {output_path}.")

    experiment_command = ["beaker", "experiment", "create", "--quiet", "--file", output_path]
    if args.name:
        experiment_command.append("--name")
        experiment_command.append(args.name.replace(" ", "-"))

    def resume_command(experiment_id):
        resume_daemon_path = os.path.join(os.path.dirname(__file__), "resume_daemon.py")
        return [
            # Run with python (instead of calling directly) in case the
            # executable bit wasn't preserved for some reason.
            "python3",
            resume_daemon_path,
            "--action=start",
            f"--max-resumes={args.max_resumes}",
            f"--experiment-id={experiment_id}",
        ]

    if args.dry_run:
        print("This is a dry run (--dry-run).  Launch your job with the following command:")
        print("    " + " ".join(experiment_command))
        if args.max_resumes > 0:
            print("Configure auto-resumes with the following command:")
            print("    " + " ".join(resume_command("$YOUR_EXPERIMENT_ID")))
    else:
        print("Running the experiment:")
        print("    " + " ".join(experiment_command))
        experiment_id = subprocess.check_output(experiment_command, universal_newlines=True).strip()
        print(
            f"Experiment {experiment_id} submitted. "
            f"See progress at https://beaker.org/ex/{experiment_id}"
        )
        if args.max_resumes > 0:
            print("Configuring auto-resumes:")
            print("    " + " ".join(resume_command(experiment_id)))
            subprocess.run(resume_command(experiment_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("param_file", type=str, help="The model configuration file.")
    parser.add_argument("--name", type=str, help="A name for the experiment.")
    parser.add_argument(
        "--spec_output_path", type=str, help="The destination to write the experiment spec."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="If specified, an experiment will not be created."
    )
    parser.add_argument(
        "--image", type=str, help="The image to use (if unspecified one will be built)"
    )
    parser.add_argument("--desc", type=str, help="A description for the experiment.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Set environment variables (e.g. NAME=value or NAME)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Bind a remote data source (e.g. source-id:/target/path)",
    )
    parser.add_argument("--cpu", help="CPUs to reserve for this experiment (e.g., 0.5)")
    parser.add_argument(
        "--gpu-count", default=1, help="GPUs to use for this experiment (e.g., 1 (default))"
    )
    parser.add_argument("--memory", help="Memory to reserve for this experiment (e.g., 1GB)")
    parser.add_argument(
        "--preemptible", action="store_true", help="Allow task to run on preemptible hardware"
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=0,
        help="When running with --preemptible, use a cronjob to automatically resume this many times.",
    )
    parser.add_argument(
        "--include-package",
        type=str,
        action="append",
        default=[],
        help="Additional packages to include",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        type=str,
        default="",
        help="a JSON structure used to override the experiment configuration",
    )

    args = parser.parse_args()

    if args.max_resumes > 0:
        assert args.preemptible, "--max-resumes requires --preemptible!"

    main(args.param_file, args)
