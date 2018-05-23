#! /usr/bin/env python
"""
NLVR dataset has at most four worlds corresponding to each sentence (with 93% of the sentences
appearing with four worlds), identified by the prefixes in identifiers. This script groups the
worlds and corresponding labels together to enable training a parser with this information.
"""

# pylint: disable=invalid-name
import json
import argparse
from collections import defaultdict


def group_dataset(input_file: str, output_file: str) -> None:
    instance_groups = defaultdict(lambda: {"worlds": [], "labels": []})
    for line in open(input_file):
        data = json.loads(line)
        # "identifier" in the original dataset looks something like 4055-3, where 4055 is common
        # across all four instances with the same sentence, but different worlds, and the suffix
        # differentiates among the four instances.
        identifier = data["identifier"].split("-")[0]
        instance_groups[identifier]["identifier"] = identifier
        instance_groups[identifier]["sentence"] = data["sentence"]
        instance_groups[identifier]["worlds"].append(data["structured_rep"])
        instance_groups[identifier]["labels"].append(data["label"])

    with open(output_file, "w") as output:
        for instance_group in instance_groups.values():
            json.dump(instance_group, output)
            output.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="NLVR data file in json format")
    parser.add_argument("output_file", type=str, help="Grouped output file in json format")
    args = parser.parse_args()
    group_dataset(args.input_file, args.output_file)
