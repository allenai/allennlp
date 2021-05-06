import dataclasses
from typing import Set

import datasets

from allennlp.steps.dataset import AllenNlpDataset
from allennlp.steps.step import Step


@Step.register("huggingface_dataset")
class HuggingfaceDataset(Step):
    DETERMINISTIC = True
    VERSION = "001"

    def run(self, dataset_name: str) -> AllenNlpDataset:
        return AllenNlpDataset(datasets.load_dataset(dataset_name), None, {"source": "huggingface"})


@Step.register("text_only")
class TextOnlyDataset(Step):
    DETERMINISTIC = True

    def run(self, input: AllenNlpDataset, fields_to_keep: Set[str]) -> AllenNlpDataset:
        return dataclasses.replace(
            input,
            splits={
                split_name: [
                    {"text": field_value}
                    for instance in split
                    for field_name, field_value in instance.items()
                    if field_name in fields_to_keep
                ]
                for split_name, split in input.splits.items()
            },
        )
