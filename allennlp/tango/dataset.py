"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Any, Optional, Sequence, Dict

from allennlp.data import Vocabulary, DatasetReader, Instance
from allennlp.tango.step import Step
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
    This step creates an `AllenNlpDataset` from old-school dataset readers. If you're
    tempted to write a new `DatasetReader`, and then use this step with it, don't.
    Just write a `Step` that creates the `AllenNlpDataset` you need directly.
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
