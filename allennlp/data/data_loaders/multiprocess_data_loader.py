from collections import deque
import logging
from multiprocessing.process import BaseProcess
import random
import traceback
from typing import List, Iterator, Optional, Iterable, Union, TypeVar

from overrides import overrides
import torch
import torch.multiprocessing as mp

from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.data_collator import DataCollator, DefaultDataCollator
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo, DatasetReaderInput
from allennlp.data.fields import TextField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util


logger = logging.getLogger(__name__)


_T = TypeVar("_T")


@DataLoader.register("multiprocess")
class MultiProcessDataLoader(DataLoader):
    """
    The `MultiProcessDataLoader` is a [`DataLoader`](../data_loader/#dataloader)
    that's optimized for AllenNLP experiments.

    See
    [Using your reader with multi-process or distributed data loading](/api/data/dataset_readers/dataset_reader/#datasetreader.using_your_reader_with_multi-process_or_distributed_data_loading)
    for more information on how to optimize your `DatasetReader` for use with this `DataLoader`.

    # Parameters

    reader: `DatasetReader`, required
        A `DatasetReader` used to load instances from the `data_path`.

    data_path: `DatasetReaderInput`, required
        Passed to `DatasetReader.read()`.

        !!! Note
            In a typical AllenNLP configuration file, the `reader` and `data_path` parameters don't
            get an entry under the `data_loader`. The `reader` is constructed separately from
            the corresponding `dataset_reader` params, and the `data_path` is taken from the
            `train_data_path`, `validation_data_path`, or `test_data_path`.

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
        The number of workers to use to read `Instances` in parallel.
        If `num_workers = 0`, everything is done in the main process. Otherwise `num_workers`
        workers are forked or spawned (depending on the value of `start_method`), each of which
        calls `read()` on their copy of the `reader`.

        This means that in order for multi-process loading to be efficient when `num_workers > 1`,
        the `reader` needs to implement
        [`manual_multiprocess_sharding`](/api/data/dataset_readers/dataset_reader/#datasetreader).

        !!! Warning
            Multi-processing code in Python is complicated! We highly recommend you read the short
            [Best practices](#multiprocessdataloader.best_practices) and
            [Common issues](#multiprocessdataloader.common_issues) sections below before using this option.

    max_instances_in_memory: `int`, optional (default = `None`)
        If not specified, all instances will be read and cached in memory for the duration
        of the data loader's life. This is generally ideal when your data can fit in memory
        during training. However, when your datasets are too big, using this option
        will turn on lazy loading, where only `max_instances_in_memory` instances are processed
        at a time.

        !!! Note
            This setting will affect how a `batch_sampler` is applied. If
            `max_instances_in_memory` is `None`, the sampler will be applied to all `Instances`.
            Otherwise the sampler will be applied to only `max_instances_in_memory` `Instances`
            at a time.

            Therefore when using this option with a sampler, you should generally set it to a multiple of
            the sampler's `batch_size` (if it has one).

    start_method: `str`, optional (default = `"fork"`)
        The [start method](https://docs.python.org/3.7/library/multiprocessing.html#contexts-and-start-methods)
        used to spin up workers.

        On Linux or OS X, "fork" usually has the lowest overhead for starting workers
        but could potentially lead to dead-locks if you're using lower-level libraries that are not fork-safe.

        If you run into these issues, try using "spawn" instead.

    cuda_device: `Optional[Union[int, str, torch.device]]`, optional (default = `None`)
        If given, batches will automatically be put on this device.

        !!! Note
            This should typically not be set in an AllenNLP configuration file. The `Trainer`
            will automatically call [`set_target_device()`](#set_target_device) before iterating
            over batches.

    quiet : `bool`, optional (default = `False`)
        If `True`, tqdm progress bars will be disabled.

    collate_fn : `DataCollator`, optional ( default = `DefaultDataCollator`)

    # Best practices

    - **Large datasets**

        If your dataset is too big to fit into memory (a common problem), you'll need to load it lazily.
        This is done by simply setting the `max_instances_in_memory` parameter to a non-zero integer.
        The optimal value depends on your use case.

        If you're using a `batch_sampler`, you will generally get better samples by setting
        `max_instances_in_memory` to a higher number - such as 10 to 100 times your batch size -
        since this determines how many `Instances` your `batch_sampler` gets to sample from at a time.

        If you're not using a `batch_sampler` then this number is much less important. Setting it to
        2 to 10 times your batch size is a reasonable value.

        Keep in mind that using `max_instances_in_memory` generally results in a slower
        training loop unless you load data in worker processes by setting the `num_workers` option to a
        non-zero integer (see below). That way data loading won't block the main process.

    - **Performance**

        The quickest way to increase the performance of data loading is adjust the `num_workers` parameter.
        `num_workers` determines how many workers are used to read `Instances` from your
        `DatasetReader`. By default, this is set to `0`, which means everything is done in the main process.

        Before trying to set `num_workers` to a non-zero number, you should make sure your `DatasetReader`
        is [optimized for use with multi-process data loading]
        (/api/data/dataset_readers/dataset_reader/#datasetreader.using_your_reader_with_multi-process_or_distributed_data_loading).

    # Common issues

    - **Dead-locks**

        Multiprocessing code in Python is complicated! Especially code that involves lower-level libraries
        which may be spawning their own threads. If you run into dead-locks while
        using `num_workers > 0`, luckily there are two simple work-arounds which usually fix the issue.

        The first work-around is to disable parallelism for these low-level libraries.
        For example, setting the environment variables `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=0`
        will do so for PyTorch and Numpy (for CPU operations) and HuggingFace Tokenizers, respectively.

        Alternatively, changing the `start_method` to "spawn" (when available, depending on your OS)
        may fix your issues without disabling parallelism for other libraries.

        See [issue #4848](https://github.com/allenai/allennlp/issues/4848) for more info.

        Dead-locks could also be caused by running out of shared memory (see below).

    - **Shared memory restrictions**

        Tensors are passed between processes using shared memory, and some systems impose strict
        limits on the allowed size of shared memory.

        Luckily this is simple to debug and simple to fix.

        First, to verify that this is your issue just watch your shared memory as your data loader runs.
        For example, run `watch -n 0.3 'df -h | grep shm'`.

        If you're seeing your shared memory blow up until it maxes-out, then you either need to decrease
        `max_instances_in_memory` or increase your system's `ulimit`.

        If you're using Docker, you can increase the shared memory available on a container by running
        it with the option `--ipc=host` or by setting `--shm-size`.

        See [issue #4847](https://github.com/allenai/allennlp/issues/4847) for more info.

    """  # noqa: E501

    def __init__(
        self,
        reader: DatasetReader,
        data_path: DatasetReaderInput,
        *,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: BatchSampler = None,
        batches_per_epoch: int = None,
        num_workers: int = 0,
        max_instances_in_memory: int = None,
        start_method: str = "fork",
        cuda_device: Optional[Union[int, str, torch.device]] = None,
        quiet: bool = False,
        collate_fn: DataCollator = DefaultDataCollator(),
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
        self.quiet = quiet
        self.cuda_device: Optional[torch.device] = None
        if cuda_device is not None:
            if not isinstance(cuda_device, torch.device):
                self.cuda_device = torch.device(cuda_device)
            else:
                self.cuda_device = cuda_device

        # Can only initialize CUDA in workers when these `start_methods` are used.
        self._worker_cuda_safe = self.start_method in {"spawn", "forkserver"}

        # To make sure we have some backpressure in the worker queues we try to set
        # reasonable defaults for the maximum size of these queues.
        # They have to be big enough that is doesn't hurt performance, but small enough
        # that they don't take up too many resources when there is a bottleneck on the
        # consuming end of a queue.
        effective_batch_size = (
            self.batch_size if self.batch_sampler is None else self.batch_sampler.get_batch_size()
        )
        self._max_instance_queue_size = (
            None
            if max_instances_in_memory is None
            else 2 * self.num_workers * max_instances_in_memory
        )
        self._max_batch_queue_size = (
            None
            if max_instances_in_memory is None
            else 2 * self.num_workers * max_instances_in_memory // (effective_batch_size or 1)
        )

        # If max_instances_in_memory is not given, we'll keep a cache of all instances in this list.
        self._instances: Optional[List[Instance]] = None
        # Keeps track of state when `batches_per_epoch` is used.
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        # For indexing instances.
        self._vocab: Optional[Vocabulary] = None

        if self.max_instances_in_memory is None:
            # Load all instances right away.
            deque(self.iter_instances(), maxlen=0)

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    @overrides
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

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        if self._vocab is None:
            raise ValueError(
                "This DataLoader has not been indexed with a Vocabulary yet. "
                "Did you forget to call DataLoader.index_with(vocab)?"
            )

        if self.batches_per_epoch is None:
            yield from self._iter_batches()
        else:
            if self._batch_generator is not None:
                batch_generator = self._batch_generator
                # Can't have a pointer to this in `self` when we try to spawn workers.
                self._batch_generator = None
            else:
                batch_generator = self._iter_batches()
            for i in range(self.batches_per_epoch):
                try:
                    yield next(batch_generator)
                except StopIteration:  # batch_generator is exhausted
                    batch_generator = self._iter_batches()  # so refresh it
                    yield next(batch_generator)
            self._batch_generator = batch_generator

    @overrides
    def iter_instances(self) -> Iterator[Instance]:
        if self._instances:
            yield from self._instances
        else:
            if self.max_instances_in_memory is None:
                self._instances = []

            if self.num_workers <= 0:
                # Just read all instances in main process.
                for instance in self._maybe_tqdm(
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
                queue: mp.JoinableQueue = (
                    ctx.JoinableQueue()
                    if self._max_instance_queue_size is None
                    else ctx.JoinableQueue(maxsize=self._max_instance_queue_size)
                )
                workers = self._start_instance_workers(queue, ctx)

                try:
                    for instance in self._maybe_tqdm(
                        self._gather_instances(queue), desc="loading instances"
                    ):
                        if self.max_instances_in_memory is None:
                            self._instances.append(instance)  # type: ignore
                        yield instance
                finally:
                    if hasattr(queue, "close"):  # for compat with different Python versions.
                        queue.close()  # type: ignore[attr-defined]
                    self._join_workers(workers, queue)

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self._instances is not None or self.num_workers <= 0:
            for batch in self._instances_to_batches(self.iter_instances(), move_to_device=True):
                yield batch
        else:
            ctx = mp.get_context(self.start_method)

            queue: mp.JoinableQueue = (
                ctx.JoinableQueue()
                if self._max_batch_queue_size is None
                else ctx.JoinableQueue(maxsize=self._max_batch_queue_size)
            )
            workers = self._start_batch_workers(queue, ctx)

            try:
                # We can now start consuming from the `queue` as the batch workers
                # produce batches.
                done_count: int = 0
                while done_count < self.num_workers:
                    for batch, worker_error in iter(queue.get, (None, None)):
                        if worker_error is not None:
                            e, tb = worker_error
                            raise WorkerError(e, tb)

                        if not self._worker_cuda_safe and self.cuda_device is not None:
                            # Need to move batch to target device now.
                            batch = nn_util.move_to_device(batch, self.cuda_device)
                        yield batch
                        queue.task_done()
                    done_count += 1
            finally:
                if hasattr(queue, "close"):  # for compat with different Python versions.
                    queue.close()  # type: ignore[attr-defined]
                self._join_workers(workers, queue)

    def _start_instance_workers(self, queue: mp.JoinableQueue, ctx) -> List[BaseProcess]:
        workers: List[BaseProcess] = []
        for worker_id in range(self.num_workers):
            worker: BaseProcess = ctx.Process(
                target=self._instance_worker, args=(worker_id, queue), daemon=True
            )
            worker.start()
            workers.append(worker)
        return workers

    def _start_batch_workers(self, queue: mp.JoinableQueue, ctx) -> List[BaseProcess]:
        workers: List[BaseProcess] = []
        for worker_id in range(self.num_workers):
            worker: BaseProcess = ctx.Process(
                target=self._batch_worker, args=(worker_id, queue), daemon=True
            )
            worker.start()
            workers.append(worker)
        return workers

    def _join_workers(self, workers: List[BaseProcess], queue) -> None:
        # Each worker will be blocking on a call to `queue.join()`,
        # calling `queue.task_done()` times the number of workers will
        # call the `queue.join()` to return, and each worker should exit on its own.
        for _ in range(len(workers)):
            try:
                queue.task_done()
            except ValueError:
                # This happens if a worker died early.
                break
        # If for some reason the workers don't exit properly, we go through and terminate
        # them anyway.
        for worker in workers:
            if worker.is_alive():
                worker.terminate()

    def _instance_worker(self, worker_id: int, queue: mp.JoinableQueue) -> None:
        try:
            self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))
            instances = self.reader.read(self.data_path)
            checked_for_token_indexers: bool = False
            for instance in instances:
                # Check the first instance to make sure it doesn't contain any TextFields with
                # token_indexers because we don't want to be duplicating those by sending
                # them across processes.
                if not checked_for_token_indexers:
                    for field_name, field in instance.fields.items():
                        if isinstance(field, TextField) and field._token_indexers is not None:
                            raise ValueError(
                                f"Found a TextField ({field_name}) with token_indexers already "
                                "applied, but you're using num_workers > 0 in your data loader. "
                                "Make sure your dataset reader's text_to_instance() method doesn't "
                                "add any token_indexers to the TextFields it creates. Instead, the token_indexers "
                                "should be added to the instances in the apply_token_indexers() method of your "
                                "dataset reader (which you'll have to implement if you haven't done "
                                "so already)."
                            )
                    checked_for_token_indexers = True
                queue.put((instance, None))
        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))

        # Indicate to the consumer that this worker is finished.
        queue.put((None, None))

        # Wait until this process can safely exit.
        queue.join()

    def _batch_worker(self, worker_id: int, queue: mp.JoinableQueue) -> None:
        try:
            self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))
            instances = self.reader.read(self.data_path)
            for batch in self._instances_to_batches(
                instances, move_to_device=self._worker_cuda_safe
            ):
                queue.put((batch, None))
        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))

        # Indicate to the consumer (main thread) that this worker is finished.
        queue.put((None, None))

        # Wait until this process can safely exit.
        queue.join()

    def _gather_instances(self, queue: mp.JoinableQueue) -> Iterable[Instance]:
        done_count: int = 0
        while done_count < self.num_workers:
            for instance, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                self.reader.apply_token_indexers(instance)
                if self._vocab is not None:
                    instance.index_fields(self._vocab)
                yield instance
                queue.task_done()
            done_count += 1

    def _index_instance(self, instance: Instance) -> Instance:
        self.reader.apply_token_indexers(instance)
        assert self._vocab is not None
        instance.index_fields(self._vocab)
        return instance

    def _instances_to_batches(
        self, instance_iterator: Iterable[Instance], move_to_device
    ) -> Iterator[TensorDict]:
        instance_iterator = (self._index_instance(instance) for instance in instance_iterator)

        if move_to_device and self.cuda_device is not None:
            tensorize = lambda batch: nn_util.move_to_device(  # noqa: E731
                self.collate_fn(batch), self.cuda_device
            )
        else:
            tensorize = self.collate_fn

        if self.batch_sampler is not None:
            instance_chunks: Iterable[List[Instance]]

            if self.max_instances_in_memory is not None:
                instance_chunks = lazy_groups_of(instance_iterator, self.max_instances_in_memory)
            else:
                instance_chunks = [list(instance_iterator)]

            for instances in instance_chunks:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
                for batch in batches:
                    yield tensorize(batch)
        else:
            # Safe to assume this is not `None` when `self.batch_sampler` is `None`.
            assert self.batch_size is not None

            if self.shuffle:
                if self.max_instances_in_memory is not None:
                    instance_iterator = shuffle_iterable(
                        instance_iterator,
                        self.max_instances_in_memory,
                    )
                else:
                    # At this point we've already loaded the instances in memory and indexed them,
                    # so this won't take long.
                    instance_iterator = list(instance_iterator)
                    random.shuffle(instance_iterator)

            for batch in lazy_groups_of(instance_iterator, self.batch_size):
                if self.drop_last and len(batch) < self.batch_size:
                    break
                yield tensorize(batch)

    def _maybe_tqdm(self, iterator: Iterable[_T], **tqdm_kwargs) -> Iterable[_T]:
        if self.quiet:
            return iterator
        return Tqdm.tqdm(iterator, **tqdm_kwargs)


class WorkerError(Exception):
    """
    An error raised when a worker fails.
    """

    def __init__(self, original_err_repr: str, traceback: List[str]) -> None:
        super().__init__(
            f"worker raised {original_err_repr}\n\n"
            "  Traceback from worker:\n  " + "".join(traceback)
            # Remove the first line of the traceback since it's redundant.
            .replace("Traceback (most recent call last):\n", "")
            # Give a little indentation so it's clear this traceback is separate from the traceback
            # in the main process.
            .replace("\n", "\n  ")
        )
