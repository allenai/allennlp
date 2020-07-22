from typing import List, Iterator

from torch.utils import data

from allennlp.common.lazy import Lazy
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo
from allennlp.data.samplers import PyTorchSampler, PyTorchBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders.data_loader import DataLoader, allennlp_collate


class AllennlpDataset(data.Dataset):
    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx) -> Instance:
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        """
        Even though it's not necessary to implement this because Python can infer
        this method from `__len__` and `__getitem__`, this helps with type-checking
        since `AllennlpDataset` can be considered an `Iterable[Instance]`.
        """
        yield from self.instances

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


class AllennlpLazyDataset(data.IterableDataset):
    """
    # Parameters

    instance_generator : `Callable[[str], Iterable[Instance]]`
        A factory function that creates an iterable of `Instance`s from a file path.
        This is usually just `DatasetReader._instance_iterator`.
    file_path : `str`
        The path to pass to the `instance_generator` function.
    vocab : `Vocab`, optional (default = `None`)
        An optional vocab. This can also be set later with the `.index_with` method.
    """

    def __init__(self, reader: DatasetReader, file_path: str, vocab: Vocabulary = None) -> None:
        super().__init__()
        self.reader = reader
        self.file_path = file_path
        self.vocab = vocab

    def __iter__(self) -> Iterator[Instance]:
        for instance in Tqdm.tqdm(self.reader.read(self.file_path), desc="reading instances"):
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


def allennlp_worker_init_fn(worker_id):
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, AllennlpLazyDataset):
        dataset.reader._set_worker_info(
            WorkerInfo(num_workers=worker_info.num_workers, worker_id=worker_id)
        )


@DataLoader.register("pytorch_data_loader", constructor="from_partial_objects")
class PyTorchDataLoader(data.DataLoader, DataLoader):
    """
    A registrable version of the pytorch
    [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
    You can use this class directly in python code, but it is identical to using
    pytorch data_loader with allennlp's custom collate and worker init functions:

    ```
    from torch.utils.data import DataLoader

    from allennlp.data.data_loaders import allennlp_collate, allennlp_worker_init_fn
    # Construct a data_loader directly for a dataset which contains allennlp
    # Instances which have _already_ been indexed.
    my_loader = PyTorchDataLoader(
        dataset,
        batch_size=32,
        collate_fn=allennlp_collate,
        worker_init_fn=allennlp_worker_init_fn,
    )
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
        sampler: PyTorchSampler = None,
        batch_sampler: PyTorchBatchSampler = None,
        num_workers: int = 0,
        # NOTE: The defaults for collate_fn and worker_init_fn are different
        # from the normal `None`. We assume that if you are using this class
        # you are using an allennlp dataset of instances, which would require these.
        collate_fn=allennlp_collate,
        worker_init_fn=allennlp_worker_init_fn,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
    ):
        super().__init__(
            dataset,
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
            # NOTE: since torch's DataLoader is listed as the first super class of this class,
            # super().__iter__() will resolve to the __iter__ method from torch's DataLoader,
            # which is what we want.
            yield from super().__iter__()
        else:
            for i in range(self._batches_per_epoch):
                try:
                    yield next(self._data_generator)
                except StopIteration:  # data_generator is exhausted
                    self._data_generator = super().__iter__()  # so refresh it
                    yield next(self._data_generator)  # and yield required instance

    def iter_instances(self) -> Iterator[Instance]:
        yield from self.dataset

    def index_with(self, vocab: Vocabulary):
        self.dataset.index_with(vocab)  # type: ignore

    @classmethod
    def from_partial_objects(
        cls,
        reader: DatasetReader,
        data_path: str,
        lazy: bool = False,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Lazy[PyTorchSampler] = None,
        batch_sampler: Lazy[PyTorchBatchSampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        multiprocessing_context: str = None,
        batches_per_epoch: int = None,
    ) -> "PyTorchDataLoader":
        dataset: data.Dataset

        if lazy:
            dataset = AllennlpLazyDataset(reader, data_path)
        else:
            dataset = AllennlpDataset(
                list(Tqdm.tqdm(reader.read(data_path), desc="reading instances"))
            )

        if sampler is not None:
            sampler_ = sampler.construct(data_source=dataset)
        else:
            sampler_ = None

        if batch_sampler is not None:
            batch_sampler_ = batch_sampler.construct(
                data_source=dataset, sampler=sampler_, batch_size=batch_size
            )
        else:
            batch_sampler_ = None

        return cls(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_,
            batch_sampler=batch_sampler_,
            num_workers=num_workers,
            # NOTE: The defaults for collate_fn and worker_init_fn are different
            # from the normal `None`. We assume that if you are using this class
            # you are using an allennlp dataset of instances, which would require these.
            collate_fn=allennlp_collate,
            worker_init_fn=allennlp_worker_init_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            multiprocessing_context=multiprocessing_context,
            batches_per_epoch=batches_per_epoch,
        )