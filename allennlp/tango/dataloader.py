"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import logging
from math import floor, ceil
from typing import Optional, Iterator, Sequence, Dict, Any

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
from allennlp.common.sequences import ShuffledSequence


class TangoDataLoader(Registrable):
    """A Tango data loader in AllenNLP is anything that produces an iterator of batches.
    You would usually initialize a data loader with a `Sequence[Instance]` to do this, but
    some Tango data loaders can be initialized in other ways and still produce batches."""

    default_implementation = "batch_size"

    def num_batches_per_epoch(self) -> Optional[int]:
        """If the dataloader produces epochs of equal length, this is how you get the length."""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[TensorDict]:
        """Override this function in your own data loader to make batches."""
        raise NotImplementedError()

    def __len__(self) -> Optional[int]:
        logging.warning(
            "This function is deprecated because it's unclear which length you get back. Please call "
            "TangoDataLoader.num_batches_per_epoch() instead."
        )
        return self.num_batches_per_epoch()


class DataLoaderAdapter(DataLoader):
    """Adapts a TangoDataLoader to an old-school AllenNLP DataLoader."""

    def __init__(self, *, tango_data_loader: TangoDataLoader):
        self.tango_data_loader = tango_data_loader
        self.target_device: Optional[torch.device] = None

    def _to_params(self) -> Dict[str, Any]:
        return {"tango_data_loader": self.tango_data_loader}

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
        raise NotImplementedError(
            "In AllenNLP Tango, your instances have to be indexed before they get to the data loader."
        )

    def set_target_device(self, device: torch.device) -> None:
        self.target_device = device


@TangoDataLoader.register("batch_size")
class BatchSizeDataLoader(TangoDataLoader):
    """A data loader that turns instances into batches with a constant number of instances
    per batch.

    * `instances` contains the instances we want to make batches out of.
    * `batch_size` is the number of instances per batch
    * `drop_last` specifies whether to keep the last batch in case it is smaller than
      `batch_size
    * `shuffle` specifies whether to shuffle the instances before making batches"""

    def __init__(
        self,
        instances: Sequence[Instance],
        *,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.instances = instances
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def _to_params(self) -> Dict[str, Any]:
        return {
            # We're not returning instances here.
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
        }

    def num_batches_per_epoch(self) -> Optional[int]:
        batch_count = len(self.instances) / self.batch_size
        if self.drop_last:
            return floor(batch_count)
        else:
            return ceil(batch_count)

    def __iter__(self) -> Iterator[TensorDict]:
        instances: Sequence[Instance]
        if self.shuffle:
            instances = ShuffledSequence(self.instances)
        else:
            instances = self.instances

        for batch in more_itertools.chunked(instances, self.batch_size):
            if not self.drop_last or len(batch) >= self.batch_size:
                yield allennlp_collate(batch)


@TangoDataLoader.register("sampler")
class SamplerDataLoader(TangoDataLoader):
    """This dataloader uses a `BatchSampler` to make batches out of the instances given in `instances`."""

    def __init__(self, instances: Sequence[Instance], *, batch_sampler: BatchSampler):
        self.instances = instances
        self.batch_sampler = batch_sampler

    def _to_params(self) -> Dict[str, Any]:
        return {"batch_sampler": self.batch_sampler}

    def num_batches_per_epoch(self) -> Optional[int]:
        return self.batch_sampler.get_num_batches(self.instances)

    def __iter__(self) -> Iterator[TensorDict]:
        for batch_indices in self.batch_sampler.get_batch_indices(self.instances):
            yield allennlp_collate([self.instances[i] for i in batch_indices])


@TangoDataLoader.register("batches_per_epoch")
class BatchesPerEpochDataLoader(TangoDataLoader):
    """This dataloader wraps another data loader, but changes the length of the epoch. It ends
    one epoch and starts another every `batches_per_epoch` batches."""

    def __init__(self, *, inner: TangoDataLoader, batches_per_epoch: int):
        self.inner = inner
        self.iter = iter(inner)
        self.batches_per_epoch = batches_per_epoch

    def _to_params(self) -> Dict[str, Any]:
        return {"inner": self.inner, "batches_per_epoch": self.batches_per_epoch}

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
    """This dataloader wraps another data loader, but only returns the first
    `max_batches_per_epoch` batches for every epoch. This is useful for debugging."""

    def __init__(self, inner: TangoDataLoader, max_batches_per_epoch: int):
        self.inner = inner
        self.max_batches_per_epoch = max_batches_per_epoch

    def _to_params(self) -> Dict[str, Any]:
        return {"inner": self.inner, "max_batches_per_epoch": self.max_batches_per_epoch}

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
