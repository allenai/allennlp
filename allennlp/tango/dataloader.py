import logging
import random
from math import floor, ceil
from typing import Optional, Iterator, Sequence

import more_itertools
import torch

from allennlp.common import Registrable
from allennlp.data import (
    TensorDict,
    Instance,
    allennlp_collate,
    BatchSampler,
    DataLoader,
    Vocabulary,
)
from allennlp.nn.util import move_to_device


class TangoDataLoader(Registrable):
    default_implementation = "batch_size"

    def num_batches_per_epoch(self) -> Optional[int]:
        """If the dataloader produces epochs of equal length, this is how you get the length."""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[TensorDict]:
        raise NotImplementedError()

    def __len__(self) -> Optional[int]:
        logging.warning(
            "This function is deprecated because it's unclear which length you get back. Please call "
            "TangoDataLoader.num_batches_per_epoch() instead."
        )
        return self.num_batches_per_epoch()


class DataLoaderAdapter(DataLoader):
    """Adapts a TangoDataLoader to an old-school AllenNLP DataLoader."""

    def __init__(self, tango_data_loader: TangoDataLoader):
        self.tango_data_loader = tango_data_loader
        self.target_device: Optional[torch.device] = None

    def __len__(self) -> int:
        result = self.tango_data_loader.num_batches_per_epoch()
        if result is None:
            raise ValueError(
                "DataLoaderAdapter cannot be used with TangoDataLoaders "
                "that don't have a defined epoch length. Old-school AllenNLP "
                "DataLoaders don't support this feature, so we can't adapt to it."
            )
        return result

    def __iter__(self) -> Iterator[TensorDict]:
        if self.target_device is None:
            return iter(self.tango_data_loader)
        else:
            for batch in iter(self.tango_data_loader):
                yield move_to_device(batch, self.target_device)

    def iter_instances(self) -> Iterator[Instance]:
        raise NotImplementedError()

    def index_with(self, vocab: Vocabulary) -> None:
        raise NotImplementedError()

    def set_target_device(self, device: torch.device) -> None:
        self.target_device = device


@TangoDataLoader.register("batch_size")
class BatchSizeDataLoader(TangoDataLoader):
    def __init__(
        self,
        instances: Sequence[Instance],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.instances = instances
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def num_batches_per_epoch(self) -> Optional[int]:
        batch_count = len(self.instances) / self.batch_size
        if self.drop_last:
            return floor(batch_count)
        else:
            return ceil(batch_count)

    def __iter__(self) -> Iterator[TensorDict]:
        instances: Sequence[Instance]
        if self.shuffle:
            instances_as_list_just_to_make_mypy_happy = list(
                self.instances
            )  # make a new list pointing to the same instance objects
            random.shuffle(instances_as_list_just_to_make_mypy_happy)
            instances = instances_as_list_just_to_make_mypy_happy
        else:
            instances = self.instances

        for batch in more_itertools.chunked(instances, self.batch_size):
            if not self.drop_last or len(batch) >= self.batch_size:
                yield allennlp_collate(batch)


@TangoDataLoader.register("sampler")
class SamplerDataLoader(TangoDataLoader):
    def __init__(self, instances: Sequence[Instance], batch_sampler: BatchSampler):
        self.instances = instances
        self.batch_sampler = batch_sampler

    def num_batches_per_epoch(self) -> Optional[int]:
        return self.batch_sampler.get_num_batches(self.instances)

    def __iter__(self) -> Iterator[TensorDict]:
        for batch_indices in self.batch_sampler.get_batch_indices(self.instances):
            yield allennlp_collate([self.instances[i] for i in batch_indices])


@TangoDataLoader.register("batches_per_epoch")
class BatchesPerEpochDataLoader(TangoDataLoader):
    def __init__(self, inner: TangoDataLoader, batches_per_epoch: int):
        self.inner = inner
        self.iter = iter(inner)
        self.batches_per_epoch = batches_per_epoch

    def num_batches_per_epoch(self) -> Optional[int]:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[TensorDict]:
        batches_yielded = 0
        while batches_yielded < self.batches_per_epoch:
            try:
                yield next(self.iter)
                batches_yielded += 1
            except StopIteration:
                self.iter = iter(self.inner)


@TangoDataLoader.register("max_batches")
class MaxBatchesDataLoader(TangoDataLoader):
    def __init__(self, inner: TangoDataLoader, max_batches_per_epoch: int):
        self.inner = inner
        self.max_batches_per_epoch = max_batches_per_epoch

    def num_batches_per_epoch(self) -> Optional[int]:
        batches = self.inner.num_batches_per_epoch()
        if batches is None:
            return None
        else:
            return min(self.max_batches_per_epoch, batches)

    def __iter__(self) -> Iterator[TensorDict]:
        for i, batch in enumerate(iter(self.inner)):
            if i >= self.max_batches_per_epoch:
                return
            else:
                yield batch
