import dataclasses
from typing import Set, Optional, Iterable, Any

from allennlp.tango.dataset import AllenNlpDataset
from allennlp.tango.step import Step


@Step.register("text_only")
class TextOnlyDataset(Step):
    """
    This step converts a dataset into another dataset that contains only the strings from the original dataset.

    You can specify exactly which fields to keep from the original dataset (default is all of them).
    You can specify a minimum length of string to keep, to filter out strings that are too short.
    """

    DETERMINISTIC = True

    def run(  # type: ignore
        self,
        input: AllenNlpDataset,
        *,
        fields_to_keep: Optional[Set[str]] = None,
        min_length: Optional[int] = None,
    ) -> AllenNlpDataset:
        def find_nested_strings(o: Any, prefix: str = "") -> Iterable[str]:
            if isinstance(o, list) or isinstance(o, tuple):
                for i, item in enumerate(o):
                    new_prefix = f"{prefix}.{i}"
                    yield from find_nested_strings(item, new_prefix)
            elif isinstance(o, dict):
                for name, item in o.items():
                    new_prefix = f"{prefix}.{name}"
                    yield from find_nested_strings(item, new_prefix)
            elif isinstance(o, str):
                if fields_to_keep is None or prefix in fields_to_keep:
                    if min_length is None or len(o) >= min_length:
                        yield o

        return dataclasses.replace(
            input,
            splits={
                split_name: [
                    {"text": text} for instance in split for text in find_nested_strings(instance)
                ]
                for split_name, split in input.splits.items()
            },
        )
