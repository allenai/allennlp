import math
import random
from typing import Optional, List, Iterator

from overrides import overrides
import torch

from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.data_collator import DefaultDataCollator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util


@DataLoader.register("simple", constructor="from_dataset_reader")
class SimpleDataLoader(DataLoader):
    """
    A very simple `DataLoader` that is mostly used for testing.
    """

    def __init__(
        self,
        instances: List[Instance],
        batch_size: int,
        *,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        self.instances = instances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch
        self.vocab = vocab
        self.cuda_device: Optional[torch.device] = None
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        self.collate_fn = DefaultDataCollator()

    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        return math.ceil(len(self.instances) / self.batch_size)

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        if self.batches_per_epoch is None:
            yield from self._iter_batches()
        else:
            if self._batch_generator is None:
                self._batch_generator = self._iter_batches()
            for i in range(self.batches_per_epoch):
                try:
                    yield next(self._batch_generator)
                except StopIteration:  # data_generator is exhausted
                    self._batch_generator = self._iter_batches()  # so refresh it
                    yield next(self._batch_generator)

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self.shuffle:
            random.shuffle(self.instances)
        for batch in lazy_groups_of(self.iter_instances(), self.batch_size):
            tensor_dict = self.collate_fn(batch)
            if self.cuda_device is not None:
                tensor_dict = nn_util.move_to_device(tensor_dict, self.cuda_device)
            yield tensor_dict

    @overrides
    def iter_instances(self) -> Iterator[Instance]:
        for instance in self.instances:
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        for instance in self.instances:
            instance.index_fields(self.vocab)

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    @classmethod
    def from_dataset_reader(
        cls,
        reader: DatasetReader,
        data_path: str,
        batch_size: int,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
        quiet: bool = False,
    ) -> "SimpleDataLoader":
        instance_iter = reader.read(data_path)
        if not quiet:
            instance_iter = Tqdm.tqdm(instance_iter, desc="loading instances")
        instances = list(instance_iter)
        return cls(instances, batch_size, shuffle=shuffle, batches_per_epoch=batches_per_epoch)
