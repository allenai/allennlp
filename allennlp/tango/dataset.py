from dataclasses import dataclass, field
from typing import Mapping, Any, Optional, Sequence

from allennlp.data import Vocabulary


@dataclass
class AllenNlpDataset:
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
