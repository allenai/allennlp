import json
import os
import sys
from collections import defaultdict
from typing import Dict, Any, Iterable, Tuple
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

JsonDict = Dict[str, Any]

def process_dataset(data: JsonDict, split_type: str) -> Iterable[Tuple[str, JsonDict]]:

    splits = defaultdict(list)

    for example in data:
        if split_type == "query_split":
            example_split = example["query-split"]
            splits[example_split].append(example)

        else:
            sentences = example.pop("sentences")

            for sentence in sentences:
                new_example = example.copy()
                new_example["sentences"] = [sentence]
                split = sentence["question-split"]
                splits[split].append(new_example)

    for split, examples in splits.items():
        if split.isdigit():
            yield ("split_" + split + ".json", examples)
        else:
            yield (split + ".json", examples)


def main(output_directory: int, data: str) -> None:
    """
    Processes the text2sql data into the following directory structure:

    ``dataset/{query_split, question_split}/{train,dev,test}.json``

    for datasets which have train, dev and test splits, or:

    ``dataset/{query_split, question_split}/{split_{split_id}}.json``

    for datasets which use cross validation.

    The JSON format is identical to the original datasets, apart from they
    are split into separate files with respect to the split_type. This means that
    for the question split, all of the sql data is duplicated for each sentence
    which is bucketed together as having the same semantics.

    As an example, the following blob would be put "as-is" into the query split
    dataset, and split into two datasets with identical blobs for the question split,
    differing only in the "sentence" key, where blob1 would end up in the train split
    and blob2 would be in the dev split, with the rest of the json duplicated in each.
    {
        "comments": [],
        "old-name": "",
        "query-split": "train",
        "sentences": [{blob1, "question-split": "train"}, {blob2, "question-split": "dev"}],
        "sql": [],
        "variables": []
    },

    Parameters
    ----------
    output_directory : str, required.
        The output directory.
    data: str, default = None
        The path to the data director of https://github.com/jkkummerfeld/text2sql-data.
    """
    json_files = glob.glob(os.path.join(data, "*.json"))

    for dataset in json_files:
        dataset_name = os.path.basename(dataset)[:-5]
        print(f"Processing dataset: {dataset} into query and question "
              f"splits at output path: {output_directory + '/' + dataset_name}")
        full_dataset = json.load(open(dataset))
        if not isinstance(full_dataset, list):
            full_dataset = [full_dataset]

        for split_type in ["query_split", "question_split"]:
            dataset_out = os.path.join(output_directory, dataset_name, split_type)

            for split, split_dataset in process_dataset(full_dataset, split_type):
                dataset_out = os.path.join(output_directory, dataset_name, split_type)
                os.makedirs(dataset_out, exist_ok=True)
                json.dump(split_dataset, open(os.path.join(dataset_out, split), "w"), indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="process text2sql data into a more readable format.")
    parser.add_argument('--out', type=str, help='The serialization directory.')
    parser.add_argument('--data', type=str, help='The path to the text2sql data directory.')
    args = parser.parse_args()
    main(args.out, args.data)
