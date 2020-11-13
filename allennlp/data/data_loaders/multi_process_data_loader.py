from collections import deque
import logging
from multiprocessing.process import BaseProcess
import random
import sys
import traceback
from typing import List, Iterator, Optional, Callable, Iterable

import torch.multiprocessing as mp

from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo
from allennlp.data.fields import TextField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


@DataLoader.register("multi_process")
class MultiProcessDataLoader(DataLoader):
    """
    The `MultiProcessDataLoader` is a [`DataLoader`](../data_loader/#dataloader)
    that's optimized for AllenNLP experiments.
    Unlike the [`PyTorchDataLoader`](../pytorch_data_loader/#pytorchdataloader),
    it can efficiently utilize multiple workers and always allows you to use a
    [`BatchSampler`](../../samplers/batch_sampler/#batchsampler).

    See
    [Using your reader with multi-process or distributed data loading](/api/data/dataset_readers/dataset_reader/#datasetreader.using_your_reader_with_multi-process_or_distributed_data_loading)
    for more information on how to optimize your `DatasetReader`.

    # Parameters

    reader: `DatasetReader`, required
        A `DatasetReader` used to load instances from the `data_path`.

    data_path: `str`, required
        Passed to `DatasetReader.read()`.

    batch_size: `int`, optional (default = `None`)
        When `batch_sampler` is unspecified, this option can be combined with `drop_last`
        and `shuffle` to control automatic batch sampling.

    drop_last: `bool`, optional (default = `False`)
        When `batch_sampler` is unspecified, this option can be combined with `batch_size`
        and `shuffle` to control automatic batch sampling.

        If `True`, the last batch will be dropped it doesn't contain a full `batch_size`
        number of `Instance`s.

    shuffle: `bool`, optional (default = `False`)
        When `batch_sampler` is unspecified, this option can be combined with `batch_size`
        and `drop_last` to control automatic batch sampling.

    batch_sampler: `BatchSampler`, optional (default = `None`)
        A `BatchSampler` to handle batching. This option is mutually exclusive with
        `batch_size`, `drop_last`, and `shuffle`.

    batches_per_epoch: `int`, optional (default = `None`)
        If specified, exactly `batches_per_epoch` batches will be generated with each call
        to `__iter__()`.

    num_workers: `int`, optional (default = `0`)
        The number of workers to use to read `Instance`s in parallel.
        If `num_workers = 0`, everything is done in the main process. Otherwise `num_workers`
        workers are forked or spawned (depending on the value of `start_method`), each of which
        calls `read()` on their copy of the `reader`.

        This means that in order for multi-process loading to be efficient when `num_workers > 1`,
        the `reader` needs to implement
        [`manual_multi_process_sharding`](/api/data/dataset_readers/dataset_reader/#datasetreader).

    collate_fn: `Callable[[List[Instance]], TensorDict]`, optional (default = `allennlp_collate`)
        The function used to turn `Instance`s into a `TensorDict` batch.

    max_instances_in_memory: `int`, optional (default = `None`)
        If not specified, all instances will be read and cached in memory for the duration
        of the data loader's life. This is generally ideal when your data can fit in memory
        during training. However, when your datasets are too big, using this option
        will turn on lazy loading, where only `max_instances_in_memory` instances are processed
        at a time.

        Note that this setting will affect how a `batch_sampler` is applied. If
        `max_instances_in_memory` is `None`, the sampler will be applied to all `Instance`s.
        Otherwise the sampler will be applied to only `max_instances_in_memory` `Instance`s
        at a time.

    start_method: `str`, optional (default = `"fork"`)
        The [start method](https://docs.python.org/3.7/library/multiprocessing.html#contexts-and-start-methods)
        used to spin up workers.

    instance_queue_size: `int`, optional (default = `1000`)
        This determines the size of the
        [multiprocessing queue](https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.Queue)
        used to pass `Instance`s from worker processes to the consuming process when
        `num_workers > 0`.

        The default value should be reasonable for most use-cases.

    instance_chunk_size: `int`, optional (default = `10`)
        This determines how many `Instance`s are sent through the instance queue
        at a time when `num_workers > 0`.

        The default value should be reasonable for most use-cases.

    batch_queue_size: `int`, optional (default = `1000`)
        This determines the size of the of the
        [multiprocessing queue](https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.Queue)
        used to pass `TensorDict` batches to the consuming process when `num_workers > 0`.

        The default value should be reasonable for most use-cases.

    batch_chunk_size: `int`, optional (default = `10`)
        This determines how many `TensorDict`s are sent through the batch queue at a time
        when `num_workers > 0`.

        The default value should be reasonable for most use-cases.


    !!! Note
        In a typical AllenNLP configuration file, the `reader` and `data_path` parameters don't
        get an entry under the "data_loader". The `reader` is constructed separately from
        the corresponding `dataset_reader` params, and the `data_path` is taken from the
        `train_data_path`, `validation_data_path`, or `test_data_path`.
    """  # noqa: E501

    def __init__(
        self,
        reader: DatasetReader,
        data_path: str,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: BatchSampler = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        collate_fn: Callable[[List[Instance]], TensorDict] = allennlp_collate,
        max_instances_in_memory: int = None,
        start_method: str = "fork",
        instance_queue_size: int = 1000,
        instance_chunk_size: int = 10,
        batch_queue_size: int = 1000,
        batch_chunk_size: int = 10,
    ) -> None:
        # Do some parameter validation.
        if num_workers is not None and num_workers < 0:
            raise ValueError("num_workers cannot be a negative number")

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if batch_sampler is not None:
            if batch_size is not None:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size")

            if drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with drop_last")

            if shuffle:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle")
        elif batch_size is None:
            raise ValueError("batch_size is required when batch_sampler is not supplied")

        if batches_per_epoch is not None and batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be at least 1")

        if max_instances_in_memory is not None:
            if batch_size is not None and max_instances_in_memory < batch_size:
                raise ValueError("max_instances_in_memory must be at least batch_size")
            elif max_instances_in_memory < 1:
                raise ValueError("max_instances_in_memory must be at least 1")

        self.reader = reader
        self.data_path = data_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_sampler = batch_sampler
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.max_instances_in_memory = max_instances_in_memory
        self.start_method = start_method
        self._instance_queue_size = instance_queue_size
        self._instance_chunk_size = instance_chunk_size
        self._batch_queue_size = batch_queue_size
        self._batch_chunk_size = batch_chunk_size

        # If max_instances_in_memory is not given, we'll keep a cache of all instances in this list.
        self._instances: Optional[List[Instance]] = None
        # Keeps track of state when `batches_per_epoch` is used.
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        # For indexing instances.
        self._vocab: Optional[Vocabulary] = None

        if self.max_instances_in_memory is None:
            # Load all instances right away.
            deque(self.iter_instances(), maxlen=0)

    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        elif self.max_instances_in_memory is None:
            # We haven't read the instances yet, so we do so now, caching them as we go.
            if not self._instances:
                deque(self.iter_instances(), maxlen=0)

            if self.batch_sampler is not None:
                return self.batch_sampler.get_num_batches(self._instances)  # type: ignore

            num_instances = len(self._instances)  # type: ignore
            # We know batch_size won't be None here since `batch_sampler` is None.
            batch_size: int = self.batch_size  # type: ignore
            if self.drop_last or num_instances % batch_size == 0:
                return num_instances // batch_size
            else:
                return 1 + num_instances // batch_size
        else:
            # We can't know the number of batches for a lazy loader when batches_per_epoch
            # is not specified.
            raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
        if self._vocab is None:
            raise ValueError(
                "This DataLoader has not been indexed with a Vocabulary yet. "
                "Did you forget to call DataLoader.index_with(vocab)?"
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
            if self.max_instances_in_memory is None:
                self._instances = []

            if self.num_workers <= 0:
                # Just read all instances in main process.
                for instance in Tqdm.tqdm(
                    self.reader.read(self.data_path), desc="loading instances"
                ):
                    self.reader.apply_token_indexers(instance)
                    if self.max_instances_in_memory is None:
                        self._instances.append(instance)  # type: ignore
                    if self._vocab is not None:
                        instance.index_fields(self._vocab)
                    yield instance
            else:
                ctx = mp.get_context(self.start_method)
                queue: mp.JoinableQueue = ctx.JoinableQueue(self._instance_queue_size)
                workers = self._start_instance_workers(queue, ctx)

                try:
                    for instance in Tqdm.tqdm(
                        self._gather_instances(queue), desc="loading instances"
                    ):
                        if self.max_instances_in_memory is None:
                            self._instances.append(instance)  # type: ignore
                        yield instance
                finally:
                    self._join_workers(workers)

    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    def _start_instance_workers(self, queue: mp.JoinableQueue, ctx) -> List[BaseProcess]:
        workers: List[BaseProcess] = []
        for worker_id in range(self.num_workers):
            worker: BaseProcess = ctx.Process(
                target=self._instance_worker, args=(worker_id, queue), daemon=True
            )
            worker.start()
            workers.append(worker)
        return workers

    def _join_workers(self, workers: List[BaseProcess]) -> None:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()

    def _gather_instances(self, queue: mp.JoinableQueue) -> Iterable[Instance]:
        done_count: int = 0
        while done_count < self.num_workers:
            for instances_chunk, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    sys.stderr.write("".join(tb))
                    raise e

                for instance in instances_chunk:
                    self.reader.apply_token_indexers(instance)
                    if self._vocab is not None:
                        instance.index_fields(self._vocab)
                    yield instance
                queue.task_done()
            queue.task_done()
            done_count += 1

    def _instance_worker(self, worker_id: int, queue: mp.JoinableQueue) -> None:
        try:
            self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))

            instances = self.reader.read(self.data_path)
            checked_for_token_indexers: bool = False

            for instances_chunk in lazy_groups_of(instances, self._instance_chunk_size):
                # Check the first instance to make sure it doesn't contain any TextFields with
                # token_indexers because we don't want to be duplicating those by sending
                # them across processes.
                if not checked_for_token_indexers:
                    for field_name, field in instances_chunk[0].fields.items():
                        if isinstance(field, TextField) and field._token_indexers is not None:
                            raise ValueError(
                                f"Found a TextField ({field_name}) with token_indexers already "
                                "applied, but you're using num_workers > 0 in your data loader. "
                                "Make sure your dataset reader's text_to_instance() method doesn't "
                                "add any token_indexers to the TextFields it creates. The token_indexers "
                                "should be added to the instances in apply_token_indexers() method of your "
                                "dataset reader (which you'll have to implement if you haven't done "
                                "so already)."
                            )
                    checked_for_token_indexers = True
                queue.put((instances_chunk, None))
        except Exception as e:
            queue.put((None, (e, traceback.format_exc())))

        # Indicate to the consumer that this worker is finished.
        queue.put((None, None))

        # Wait for consumer to finish to avoid prematurely closing the queue.
        queue.join()

    def _instances_to_batches(self, instance_iterator: Iterable[Instance]) -> Iterator[TensorDict]:
        if self.max_instances_in_memory is not None:
            if self.shuffle:
                instance_iterator = shuffle_iterable(
                    instance_iterator, self.max_instances_in_memory
                )
            instance_chunks: Iterable[List[Instance]] = lazy_groups_of(
                instance_iterator, self.max_instances_in_memory
            )
        else:
            instance_chunks = [list(instance_iterator)]
            if self.shuffle:
                random.shuffle(instance_chunks[0])

        for instances in instance_chunks:
            batches: Iterator[List[Instance]]
            if self.batch_sampler:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
            else:
                # NOTE: it's safe to assume `batch_size` is not `None` when `batch_sampler` is `None`.
                # Hence the `type: ignore` comment.
                batches = lazy_groups_of(instances, self.batch_size)  # type: ignore[arg-type]

            for batch in batches:
                if (
                    self.batch_sampler is None
                    and self.drop_last
                    and len(batch) < self.batch_size  # type: ignore[operator]
                ):
                    break
                yield self.collate_fn(batch)

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self._instances is not None or self.num_workers <= 0:
            for batch in self._instances_to_batches(self.iter_instances()):
                yield batch
        else:
            ctx = mp.get_context(self.start_method)

            # First we start "instance workers", which are in charge of generating raw
            # instances using self.reader. The generated instances are then put
            # into the `instance_queue` for the `batch_worker` to consume.
            instance_queue: mp.JoinableQueue = ctx.JoinableQueue(self._instance_queue_size)
            instance_workers = self._start_instance_workers(instance_queue, ctx)

            # Now start the `batch_worker`. This worker consumes from the `instance_queue`
            # and puts the resulting batches into the `batch_queue`.
            batch_queue: mp.JoinableQueue = ctx.JoinableQueue(self._batch_queue_size)
            batch_worker: BaseProcess = ctx.Process(
                target=self._batch_worker,
                args=(
                    instance_queue,
                    batch_queue,
                ),
                daemon=True,
            )
            batch_worker.start()

            try:
                # We can now start consuming from the `batch_queue` as the `batch_worker`
                # produces batches.
                for batch_group, worker_error in iter(batch_queue.get, (None, None)):
                    if worker_error is not None:
                        e, tb = worker_error
                        sys.stderr.write("".join(tb))
                        raise e

                    yield from batch_group
                    batch_queue.task_done()

                # Indicate to the worker (producer of batch groups) that we've consumed
                # everything.
                batch_queue.task_done()
            finally:
                self._join_workers(instance_workers + [batch_worker])

    def _batch_worker(
        self, instance_queue: mp.JoinableQueue, batch_queue: mp.JoinableQueue
    ) -> None:
        try:
            for batch_chunk in lazy_groups_of(
                self._instances_to_batches(self._gather_instances(instance_queue)),
                self._batch_chunk_size,
            ):
                batch_queue.put((batch_chunk, None))
        except Exception as e:
            batch_queue.put((None, (e, traceback.format_exc())))

        # Indicate to the consumer (main thread) that this worker is finished.
        batch_queue.put((None, None))

        # Wait for the consumer (in the main process) to finish receiving all batch groups
        # to avoid prematurely closing the queue.
        batch_queue.join()
