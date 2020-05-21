from typing import List, Dict, Union

import torch
from torch.utils import data

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance

from allennlp.common.lazy import Lazy
from allennlp.data.batch import Batch
from allennlp.data.samplers import Sampler, BatchSampler


TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


def allennlp_collate(instances: List[Instance]) -> TensorDict:
    batch = Batch(instances)
    return batch.as_tensor_dict(batch.get_padding_lengths())


class DataLoader(Registrable, data.DataLoader):
    """
    A registrable version of the pytorch
    [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
    Firstly, this class exists is so that we can construct a DataLoader
    from a configuration file and have a different default `collate_fn`.
    You can use this class directly in python code, but it is identical to using
    pytorch dataloader with allennlp's custom collate function:

    ```
    from torch.utils.data import DataLoader

    from allennlp.data.samplers import allennlp_collate
    # Construct a dataloader directly for a dataset which contains allennlp
    # Instances which have _already_ been indexed.
    my_loader = DataLoader(dataset, batch_size=32, collate_fn=allennlp_collate)
    ```

    Secondly, this class adds a `batches_per_epoch` parameter which, if given, determines the number
    of batches after which an epoch ends.  If this is `None`, then an epoch is set to be one full pass
    through your data.  You might use this if you have a very large dataset and want more frequent
    checkpoints and evaluations on validation data, for instance.

    In a typical AllenNLP configuration file, the `dataset` parameter does not get an entry under
    the "data_loader", it gets constructed separately.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Sampler = None,
        batch_sampler: BatchSampler = None,
        num_workers: int = 0,
        # NOTE: The default for collate_fn is different from the normal `None`.
        # We assume that if you are using this class you are using an
        # allennlp dataset of instances, which would require this.
        collate_fn=allennlp_collate,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )
        self._data_generator = super().__iter__()
        self._batches_per_epoch = batches_per_epoch

    def __len__(self):
        if self._batches_per_epoch is not None:
            return self._batches_per_epoch
        return super().__len__()

    def __iter__(self):
        if self._batches_per_epoch is None:
            yield from super().__iter__()
        else:
            for i in range(self._batches_per_epoch):
                try:
                    yield next(self._data_generator)
                except StopIteration:  # data_generator is exhausted
                    self._data_generator = super().__iter__()  # so refresh it
                    yield next(self._data_generator)  # and yield required instance

    @classmethod
    def from_partial_objects(
        cls,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Lazy[Sampler] = None,
        batch_sampler: Lazy[BatchSampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn=None,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
    ) -> "DataLoader":

        if batch_sampler is not None:
            batch_sampler_ = batch_sampler.construct(data_source=dataset)
        else:
            batch_sampler_ = None
        if sampler is not None:
            sampler_ = sampler.construct(data_source=dataset)
        else:
            sampler_ = None

        return cls(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_,
            batch_sampler=batch_sampler_,
            num_workers=num_workers,
            # NOTE: The default for collate_fn is different from the normal `None`.
            # We assume that if you are using this class you are using an
            # allennlp dataset of instances, which would require this.
            collate_fn=allennlp_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            batches_per_epoch=batches_per_epoch,
        )


DataLoader.register("default", "from_partial_objects")(DataLoader)
DataLoader.default_implementation = "default"
