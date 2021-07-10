from typing import Set

import dataclasses

from allennlp.tango.step import Step
from allennlp.tango.dataset import AllenNlpDataset


@Step.register("keep_fields")
class KeepFields(Step):
    DETERMINISTIC = True

    def run(self, input: AllenNlpDataset, fields_to_keep: Set[str]) -> AllenNlpDataset:  # type: ignore
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
