from typing import List, Dict, Union, Iterator

import torch

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.batch import Batch
from allennlp.data.vocabulary import Vocabulary


TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def allennlp_collate(instances: List[Instance]) -> TensorDict:
    batch = Batch(instances)
    return batch.as_tensor_dict(batch.get_padding_lengths())


class DataLoader(Registrable):
    """
    A `DataLoader` is responsible for generating batches of instances from a `DatasetReader`,
    or another source of data.

    This class has three required methods:
      - `__iter__()` that creates an iterable of `TensorDict`s,
      - `iter_instances()` that creates an iterable of `Instance`s, and
      - `index_with()` that should index the data with a vocabulary.

    Additionally, this class should also implement `__len__()` when possible.

    The default implementation is `MultiProcessDataLoader`.
    """

    default_implementation = "multi_process_dataloader"

    def __len__(self) -> int:
        raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
        raise NotImplementedError

    def iter_instances(self) -> Iterator[Instance]:
        raise NotImplementedError

    def index_with(self, vocab: Vocabulary) -> None:
        raise NotImplementedError
