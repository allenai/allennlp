from typing import List, Dict, Union, Iterator

import torch

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.batch import Batch
from allennlp.data.vocabulary import Vocabulary


TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
"""
`TensorDict` is the type we use for batches.
"""


def allennlp_collate(instances: List[Instance]) -> TensorDict:
    """
    This is the default function used to turn a list of `Instance`s into a `TensorDict`
    batch.
    """
    batch = Batch(instances)
    return batch.as_tensor_dict()


class DataLoader(Registrable):
    """
    A `DataLoader` is responsible for generating batches of instances from a
    [`DatasetReader`](/api/data/dataset_readers/dataset_reader/#datasetreader),
    or another source of data.

    This class has three required methods:

      - [`__iter__()`](#__iter__) that creates an iterable of `TensorDict`s,
      - [`iter_instances()`](#iter_instances) that creates an iterable of `Instance`s, and
      - [`index_with()`](#index_with) that should index the data with a vocabulary.

    Additionally, this class should also implement `__len__()` when possible.

    The default implementation is
    [`MultiProcessDataLoader`](../multi_process_data_loader/#multiprocessdataloader).
    """

    default_implementation = "multi_process"

    def __len__(self) -> int:
        raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
        raise NotImplementedError

    def iter_instances(self) -> Iterator[Instance]:
        raise NotImplementedError

    def index_with(self, vocab: Vocabulary) -> None:
        raise NotImplementedError
