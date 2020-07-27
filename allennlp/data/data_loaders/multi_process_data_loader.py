from collections import deque
import logging
import random
from typing import List, Iterator, Optional, Callable, Iterable

import torch.multiprocessing as mp

from allennlp.common.lazy import Lazy
from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


@DataLoader.register("multi_process_data_loader", constructor="from_partial_objects")
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
        collate_fn: Callable[[List[Instance]], TensorDict] = allennlp_collate,
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
        self.collate_fn = collate_fn
        self.lazy = lazy
        self.max_batches_in_memory = max_batches_in_memory

        # When lazy = False, we'll keep a cache of all instances in this list.
        self._instances: Optional[List[Instance]] = None
        # Keeps track of state when `batches_per_epoch` is used.
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        # For indexing instances.
        self._vocab: Optional[Vocabulary] = None

        if not self.lazy:
            # Load instances right away.
            deque(self.iter_instances(), maxlen=0)

    def __len__(self) -> int:
        if not self.lazy:
            # We haven't read the instances yet, so we do so now, caching them as we go.
            if not self._instances:
                deque(self.iter_instances(), maxlen=0)

            if self.batch_sampler is not None:
                return self.batch_sampler.get_num_batches(self._instances)  # type: ignore

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
        if self._vocab is None:
            raise ValueError(
                "You must index the data loader .index_with(vocab) before generating batches"
            )
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
            yield from self._instances
        else:
            if not self.lazy:
                self._instances = []

            if self.num_workers <= 0:
                # Just read all instances in main process.
                for instance in self.reader.read(self.data_path):
                    if not self.lazy:
                        self._instances.append(instance)  # type: ignore
                    if self._vocab is not None:
                        instance.index_fields(self._vocab)
                    yield instance
            else:
                queue: mp.JoinableQueue = mp.JoinableQueue(self._INSTANCE_QUEUE_SIZE)
                workers = self._start_instance_workers(queue)

                for instance in Tqdm.tqdm(self._gather_instances(queue), desc="loading instances"):
                    if not self.lazy:
                        self._instances.append(instance)  # type: ignore
                    yield instance

                self._join_workers(workers)

    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    def _start_instance_workers(self, queue: mp.JoinableQueue) -> List[mp.Process]:
        workers: List[mp.Process] = []
        for worker_id in range(self.num_workers):
            worker = mp.Process(target=self._instance_worker, args=(worker_id, queue), daemon=True)
            worker.start()
            workers.append(worker)
        return workers

    def _join_workers(self, workers: List[mp.Process]) -> None:
        for worker in workers:
            # TODO: handle errors if any of the workers crash.
            worker.join(1)
            if worker.is_alive():
                logger.warning("Worker is still alive, killing now")
                worker.terminate()

    def _gather_instances(self, queue: mp.JoinableQueue) -> Iterable[Instance]:
        done_count: int = 0
        while done_count < self.num_workers:
            instances_chunk: Optional[List[Instance]]
            for instances_chunk in iter(queue.get, None):
                yield from instances_chunk
                queue.task_done()
            queue.task_done()
            # Every time we encounter an empty list, thats means a worker has finished.
            done_count += 1

    def _instance_worker(self, worker_id: int, queue: mp.JoinableQueue) -> None:
        self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))

        instances: Iterator[Instance]
        if self._vocab is not None:

            def index_fields(instance: Instance) -> Instance:
                instance.index_fields(self._vocab)  # type: ignore
                return instance

            instances = (index_fields(instance) for instance in self.reader.read(self.data_path))

        else:
            instances = self.reader.read(self.data_path)

        for instances_chunk in lazy_groups_of(instances, self._INSTANCE_CHUNK_SIZE):
            queue.put(instances_chunk)

        # Indicate to the consumer that this worker is finished.
        queue.put(None)

        # Wait for consumer to finish to avoid prematurely closing the queue.
        queue.join()

    def _instances_to_batches(self, instance_iterator: Iterable[Instance]) -> Iterator[TensorDict]:
        instance_chunks: Iterable[List[Instance]]
        if self.max_batches_in_memory is not None:
            chunk_size = self.batch_size * self.max_batches_in_memory
            instance_chunks = lazy_groups_of(instance_iterator, chunk_size)
        else:
            instance_chunks = [list(instance_iterator)]

        for instances in instance_chunks:
            if self.shuffle:
                random.shuffle(instances)

            batches: Iterator[List[Instance]]
            if self.batch_sampler:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
            else:
                batches = lazy_groups_of(instances, self.batch_size)

            batched_tensor_dicts = (self.collate_fn(batch) for batch in batches)

            yield from batched_tensor_dicts

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self._instances is not None or self.num_workers <= 0:
            for batch in self._instances_to_batches(self.iter_instances()):
                yield batch
        else:
            # At this point self.max_batches_in_memory is not None since lazy must be False.
            assert self.max_batches_in_memory is not None

            # First we start "instance workers", which are in charge generating raw
            # instances using self.reader. The generated instances are then put
            # into the `instance_queue` for the `batch_worker` to consume.
            instance_queue: mp.JoinableQueue = mp.JoinableQueue(self._INSTANCE_QUEUE_SIZE)
            instance_workers = self._start_instance_workers(instance_queue)

            # Now start the `batch_worker`. This worker consumes from the `instance_queue`
            # and puts the resulting batches into the `batch_queue`.
            batch_queue: mp.JoinableQueue = mp.JoinableQueue(self.max_batches_in_memory)  # type: ignore
            batch_worker = mp.Process(
                target=self._batch_worker, args=(instance_queue, batch_queue,), daemon=True
            )
            batch_worker.start()

            # We can now start consuming from the `batch_queue` as the `batch_worker`
            # produces batches.
            batch_group: Optional[List[TensorDict]]
            for batch_group in iter(batch_queue.get, None):
                yield from batch_group
                batch_queue.task_done()

            # Indicate to the worker (producer of batch groups) that we've consumed
            # everything.
            batch_queue.task_done()

            # Join all of the workers.
            self._join_workers(instance_workers + [batch_worker])

    def _batch_worker(
        self, instance_queue: mp.JoinableQueue, batch_queue: mp.JoinableQueue
    ) -> None:
        for batch_chunk in lazy_groups_of(
            self._instances_to_batches(self._gather_instances(instance_queue)),
            self._BATCH_CHUNK_SIZE,
        ):
            batch_queue.put(batch_chunk)

        # Indicate to the consumer (main thread) that this worker is finished.
        batch_queue.put(None)

        # Wait for the consumer (in the main process) to finish receiving all batch groups
        # to avoid prematurely closing the queue.
        batch_queue.join()

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
