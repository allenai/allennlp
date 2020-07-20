from collections import deque
import logging
import random
from typing import List, Iterator, Optional

import torch.multiprocessing as mp

from allennlp.common.lazy import Lazy
from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.dataloaders.dataloader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


@DataLoader.register("multi_process_dataloader", constructor="from_partial_objects")
class MultiProcessDataLoader(DataLoader):
    _INSTANCE_QUEUE_SIZE = 1000
    _INSTANCE_CHUNK_SIZE = 10
    _BATCH_CHUNK_SIZE = 10

    def __init__(
        self,
        reader: DatasetReader,
        data_path: str,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: BatchSampler = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        lazy: bool = False,
        max_batches_in_memory: int = None,
    ) -> None:
        # Do some parameter validation.
        if num_workers is not None and num_workers < 0:
            raise ValueError("num_workers cannot be a negative number")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batches_per_epoch is not None and batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be at least 1")
        if lazy:
            if max_batches_in_memory is None:
                raise ValueError("max_batches_in_memory must be specified for a lazy loader")
            if max_batches_in_memory < 1:
                raise ValueError("max_batches_in_memory must be at least 1")

        self.reader = reader
        self.data_path = data_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_sampler = batch_sampler
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers
        self.lazy = lazy
        self.max_batches_in_memory = max_batches_in_memory

        # When lazy = False, we'll keep a cache of all instances in this list.
        self._instances: Optional[List[Instance]] = None
        # Keeps track of state when `batches_per_epoch` is used.
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        # For indexing instances.
        self._vocab: Optional[Vocabulary] = None

    def __len__(self) -> int:
        if not self.lazy:
            # We haven't read the instances yet, so we do so now, caching them as we go.
            if not self._instances:
                deque(self.iter_instances(), maxlen=0)

            num_instances = len(self._instances)  # type: ignore
            if self.drop_last or num_instances % self.batch_size == 0:
                return num_instances // self.batch_size
            else:
                return 1 + num_instances // self.batch_size
        elif self.batches_per_epoch is not None:
            return self.batches_per_epoch
        else:
            # We can't know the number of batches for a lazy loader when batches_per_epoch
            # is not specified.
            raise TypeError

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

    def iter_instances(self) -> Iterator[Instance]:
        if self._instances:
            return iter(self._instances)

        if not self.lazy:
            self._instances = []

        if not self.num_workers:
            # Just read all instances in main process.
            for instance in self.reader.read(self.data_path):
                if not self.lazy:
                    self._instances.append(instance)  # type: ignore
                yield instance
        else:
            queue: mp.Queue = mp.Queue(self._INSTANCE_QUEUE_SIZE)
            workers: List[mp.Process] = []
            for worker_id in range(self.num_workers):
                worker = mp.Process(target=self._instance_worker, args=(worker_id, queue))
                worker.start()
                workers.append(worker)

            def instance_iterator():
                done_count: int = 0
                while done_count < self.num_workers:
                    for instances_chunk in iter(queue.get, []):
                        for instance in instances_chunk:
                            yield instance
                    # Every time we encounter an empty list, thats means a worker has finished.
                    done_count += 1

            for instance in Tqdm.tqdm(instance_iterator()):
                if not self.lazy:
                    self._instances.append(instance)  # type: ignore
                yield instance

            # Clean up.
            for worker_id, worker in enumerate(workers):
                # TODO: handle errors if any of the workers crash.
                worker.join(1)
                if worker.is_alive():
                    logger.info("Worker %d is still alive, killing now", worker_id)

    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    def _instance_worker(self, worker_id: int, queue: mp.Queue) -> None:
        self.reader.worker_info = WorkerInfo(num_workers=self.num_workers, worker_id=worker_id)

        if self._vocab is not None:

            def index_fields(instance: Instance) -> Instance:
                instance.index_fields(self._vocab)
                return instance

            instances = (index_fields(instance) for instance in self.reader.read(self.data_path))

        else:
            instances = self.reader.read(self.data_path)

        for instances_chunk in lazy_groups_of(instances, self._INSTANCE_CHUNK_SIZE):
            queue.put(instances_chunk)

        # Indicate to the consumer that this worker is finished.
        queue.put([])

    def _iter_batches(self) -> Iterator[TensorDict]:
        if not self.lazy and not self._instances:
            # Cache instances to `self._instances`.
            deque(self.iter_instances(), maxlen=0)

        if self._instances:
            if not self.batch_sampler and self.shuffle:
                logger.info("Shuffling instances")
                random.shuffle(self._instances)

            batches: Iterator[List[Instance]]
            if self.batch_sampler:
                batches = (
                    [self._instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(self._instances)
                )
            else:
                batches = lazy_groups_of(self._instances, self.batch_size)

            for batch in batches:
                yield allennlp_collate(batch)
        else:
            queue: mp.Queue = mp.Queue(self.max_batches_in_memory)  # type: ignore
            worker = mp.Process(target=self._batch_worker, args=(queue,))
            worker.start()

            for batch_group in iter(queue.get, []):
                yield from batch_group

            # TODO: handle errors if the worker crashes.
            worker.join()

    def _batch_worker(self, queue: mp.Queue) -> None:
        chunk_size = self.batch_size * self.max_batches_in_memory  # type: ignore
        for instances in lazy_groups_of(self.iter_instances(), chunk_size):
            if not self.batch_sampler and self.shuffle:
                random.shuffle(instances)

            batches: Iterator[List[Instance]]
            if self.batch_sampler:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
            else:
                batches = lazy_groups_of(instances, self.batch_size)

            batched_tensor_dicts = (allennlp_collate(batch) for batch in batches)

            for batch_group in lazy_groups_of(batched_tensor_dicts, self._BATCH_CHUNK_SIZE):
                queue.put(batch_group)

        # Indicate to the consumer (main thread) that this worker is finished.
        queue.put([])

    @classmethod
    def from_partial_objects(
        cls,
        reader: DatasetReader,
        data_path: str,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: Lazy[BatchSampler] = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        lazy: bool = False,
        max_batches_in_memory: int = None,
    ) -> "MultiProcessDataLoader":
        if batch_sampler is not None:
            batch_sampler_ = batch_sampler.construct(batch_size=batch_size)
        else:
            batch_sampler_ = None

        return cls(
            reader,
            data_path,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            batch_sampler=batch_sampler_,
            batches_per_epoch=batches_per_epoch,
            num_workers=num_workers,
            lazy=lazy,
            max_batches_in_memory=max_batches_in_memory,
        )
