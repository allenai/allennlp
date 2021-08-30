"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

from typing import Set, Optional, Iterable, Any

from allennlp.tango.dataset import DatasetDict
from allennlp.tango.step import Step
from allennlp.common.sqlite_sparse_sequence import SqliteSparseSequence
from allennlp.tango.sqlite_format import SqliteDictFormat
from tqdm import tqdm


@Step.register("strings_only")
class StringsOnlyDataset(Step):
    """
    This step converts a dataset into another dataset that contains only strings from the original dataset.

    You can specify exactly which fields to keep from the original dataset (default is all of them).
    You can specify a minimum length of string to keep, to filter out strings that are too short.
    """

    DETERMINISTIC = True
    VERSION = "001"
    FORMAT = SqliteDictFormat()

    def run(  # type: ignore
        self,
        input: DatasetDict,
        *,
        fields_to_keep: Optional[Set[str]] = None,
        min_length: Optional[int] = None,
    ) -> DatasetDict:
        """
        Turns the `input` dataset into another dataset that contains only the strings from the
        original dataset.

        * `fields_to_keep` is an optional list of field names that you want to keep in the result.
          If this is `None`, all fields are kept.
        * `min_length` specifies the minimum length that a string must have to be part of the
          result. If this is `None`, all strings are considered.
        """

        def find_nested_strings(o: Any, prefix: str = "") -> Iterable[str]:
            if isinstance(o, str):
                if fields_to_keep is None or prefix in fields_to_keep:
                    if min_length is None or len(o) >= min_length:
                        yield o
            elif isinstance(o, list) or isinstance(o, tuple):
                for i, item in enumerate(o):
                    new_prefix = f"{prefix}.{i}"
                    yield from find_nested_strings(item, new_prefix)
            elif isinstance(o, dict):
                for name, item in o.items():
                    new_prefix = f"{prefix}.{name}"
                    yield from find_nested_strings(item, new_prefix)

        splits = {}
        for split_name, split in input.splits.items():
            sequence_file = self.work_dir() / f"{split_name}.sqlite"
            sequence_file.unlink(missing_ok=True)
            sequence = SqliteSparseSequence(sequence_file)
            sequence.extend(
                {"text": string}
                for instance in tqdm(split, desc=f"Processing split '{split_name}'")
                for string in find_nested_strings(instance)
            )
            splits[split_name] = sequence

        return DatasetDict(splits=splits, vocab=input.vocab, metadata=input.metadata)
