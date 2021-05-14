from dataclasses import dataclass, field
from typing import Mapping, Any, Optional, Sequence

from allennlp.data import Vocabulary


@dataclass
class AllenNlpDataset:
    splits: Mapping[str, Sequence[Any]]
    vocab: Optional[Vocabulary] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
