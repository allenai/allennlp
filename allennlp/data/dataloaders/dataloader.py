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
    A `DataLoader` is responsible for generating batches of instances from a `Dataset`,
    or another source of data.

    This class only has one required method, `__iter__()`, that creates an iterable
    of `TensorDict`s.

    Additionally, this class should also implement `__len__()` when possible.
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
