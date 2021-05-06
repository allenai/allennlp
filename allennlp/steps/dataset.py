from dataclasses import dataclass, field
from typing import Mapping, Any, Sequence, Optional

from allennlp.data import Vocabulary

Split = Sequence[Any]


@dataclass
class AllenNlpDataset:
    splits: Mapping[str, Split]
    vocab: Optional[Vocabulary] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
