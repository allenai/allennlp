import logging
from torch.utils import data

from allennlp.common.registrable import Registrable

from allennlp.common.lazy import Lazy
from allennlp.data.batch import Batch
from allennlp.data.samplers.samplers import (
    Sampler,
    BatchSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    RandomSampler,
    BasicBatchSampler,
    BatchInstanceSampler,
)

logger = logging.getLogger(__name__)


def allennlp_collocate(batch):
    batch = Batch(batch)
    return batch.as_tensor_dict(batch.get_padding_lengths())


class DataLoader(Registrable, data.DataLoader):
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Lazy[Sampler] = None,
        batch_sampler: Lazy[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
    ):

        collate_fn = allennlp_collocate
        if batch_sampler is not None:
            batch_sampler_ = batch_sampler.construct(data_source=dataset)
        else:
            batch_sampler_ = None
        if sampler is not None:
            sampler_ = sampler.construct(data_source=dataset)
        else:
            sampler_ = None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_,
            batch_sampler=batch_sampler_,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )
