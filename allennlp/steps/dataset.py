from dataclasses import dataclass, field
from typing import Mapping, Any, Optional, Collection

from allennlp.data import Vocabulary, Batch


@dataclass
class AllenNlpDataset:
    splits: Mapping[str, Collection[Any]]
    vocab: Optional[Vocabulary] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
