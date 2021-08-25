"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Mapping, Any, Optional, Sequence, Dict

from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.tango.step import Step
from allennlp.common.sequences import SlicedSequence, ConcatenatedSequence, ShuffledSequence
from tqdm import tqdm


@dataclass
class DatasetDict:
    """This definition of a dataset combines all splits, the vocabulary, and some metadata, into
    one handy class."""

    splits: Mapping[str, Sequence[Any]]
    """Maps the name of the split to a sequence of instances. `AllenNlpDataset` does not
    enforce what the instances are made of, so they are of type `Sequence[Any]`. However,
    the data loader will care about the type of the instances, and surely throw an error
    if it encounters a type it cannot handle."""

    vocab: Optional[Vocabulary] = None
    """The vocabulary of this dataset."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Metadata can contain anything you need."""

    def __getitem__(self, split: str) -> Sequence[Any]:
        return self.splits[split]

    def __len__(self) -> int:
        return len(self.splits)


@Step.register("dataset_reader_adapter")
class DatasetReaderAdapterStep(Step):
    """
    This step creates an `DatasetDict` from old-school dataset readers. If you're
    tempted to write a new `DatasetReader`, and then use this step with it, don't.
    Just write a `Step` that creates the `DatasetDict` you need directly.
    """

    DETERMINISTIC = True  # We're giving the dataset readers some credit here.
    CACHEABLE = True
    VERSION = "002"

    def run(self, reader: DatasetReader, splits: Dict[str, str]) -> DatasetDict:  # type: ignore
        """
        * `reader` specifies the old-school dataset reader to use.
        * `splits` maps the names of the splits to the filenames to use for the
          dataset reader. It might look like this:
          ```
          {
              "train": "/path/to/train.json",
              "validation": "/path/to/validation.json"
          }
          ```
        """
        instances_map: Dict[str, Sequence[Instance]] = {
            split_name: list(tqdm(reader.read(path), desc=f"Reading {path}"))
            for split_name, path in splits.items()
        }
        vocab = Vocabulary.from_instances(itertools.chain(*instances_map.values()))

        # index all the instances with the vocab
        for split_name, instances in instances_map.items():
            for instance in tqdm(instances, desc=f"Indexing {split_name}"):
                instance.index_fields(vocab)

        return DatasetDict(splits=instances_map, vocab=vocab)


@Step.register("dataset_remix")
class DatasetRemixStep(Step):
    """
    This step can remix splits in a dataset into new splits.
    """

    DETERMINISTIC = True
    CACHEABLE = False  # This is so fast it's not worth caching.
    VERSION = "001"

    def run(  # type: ignore
        self,
        input: DatasetDict,
        new_splits: Dict[str, str],
        keep_old_splits: bool = True,
        shuffle_before: bool = False,
        shuffle_after: bool = False,
        random_seed: int = 1532637578,
    ) -> DatasetDict:
        random.seed(random_seed)

        if shuffle_before:
            input_splits: Mapping[str, Sequence[Any]] = {
                split_name: ShuffledSequence(split_instances)
                for split_name, split_instances in input.splits.items()
            }
        else:
            input_splits = input.splits

        def get_slice(split_name: str) -> Sequence[Any]:
            slice_match = re.match(r"(.*)\[([0123456789:]*)]", split_name)
            if slice_match is None:
                return input[split_name]
            else:
                split_name = slice_match[1]
                slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
                return SlicedSequence(input[split_name], slice(*slice_args))

        def parse_split_spec(split_spec: str):
            parts = [get_slice(name.strip()) for name in split_spec.split("+")]
            if len(parts) == 1:
                return parts[0]
            else:
                return ConcatenatedSequence(*parts)

        if keep_old_splits:
            result = dict(input_splits.items())
        else:
            result = {}
        result.update(
            {
                new_split_name: parse_split_spec(new_split_spec)
                for new_split_name, new_split_spec in new_splits.items()
            }
        )

        if shuffle_after:
            result = {
                split_name: ShuffledSequence(split_instances)
                for split_name, split_instances in result.items()
            }

        return DatasetDict(vocab=input.vocab, metadata=input.metadata, splits=result)
