import logging
import os
from queue import Empty
from typing import Iterable, Iterator, List, Optional

from torch.multiprocessing import JoinableQueue, Process, Queue, get_logger

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.multiprocess_dataset_reader import QIterable
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary

logger = get_logger()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

def _create_tensor_dicts_from_queue(input_queue: Queue,
                                    output_queue: Queue,
                                    iterator: DataIterator,
                                    shuffle: bool,
                                    index: int) -> None:
    """
    Pulls instances from ``input_queue``, converts them into ``TensorDict``s
    using ``iterator``, and puts them on the ``output_queue``.
    """
    logger.info(f"Iterator worker: {index} PID: {os.getpid()}")
    def instances() -> Iterator[Instance]:
        instance = input_queue.get()
        while instance is not None:
            yield instance
            instance = input_queue.get()

    for tensor_dict in iterator(instances(), num_epochs=1, shuffle=shuffle):
        output_queue.put(tensor_dict)

    output_queue.put(index)

    # We need to ensure we've gotten all the tensors out of this queue before
    # this process ends. Otherwise we'll crash. See
    # https://github.com/pytorch/pytorch/issues/7181. This appears to be an
    # issue specifically with tensors, perhaps due to the refcounting involved
    # in managing them in shared memory. If you're working on this code, be
    # aware that I've only been able to reproduce this issue on Linux.  Testing
    # on a Mac alone is not sufficient.
    output_queue.join()

def _create_tensor_dicts_from_qiterable(qiterable: QIterable,
                                        output_queue: Queue,
                                        iterator: DataIterator,
                                        shuffle: bool,
                                        index: int) -> None:
    """
    Pulls instances from ``qiterable.output_queue``, converts them into
    ``TensorDict``s using ``iterator``, and puts them on the ``output_queue``.
    """
    logger.info(f"Iterator worker: {index} PID: {os.getpid()}")
    def instances() -> Iterator[Instance]:
        while qiterable.num_active_workers.value > 0 or qiterable.num_inflight_items.value > 0:
            while True:
                try:
                    yield qiterable.output_queue.get(block=False, timeout=1.0)
                    with qiterable.num_inflight_items.get_lock():
                        qiterable.num_inflight_items.value -= 1
                except Empty:
                    break

    for tensor_dict in iterator(instances(), num_epochs=1, shuffle=shuffle):
        output_queue.put(tensor_dict)

    output_queue.put(index)

    # See the note above in _create_tensor_dicts_from_queue.
    output_queue.join()

def _queuer(instances: Iterable[Instance],
            input_queue: Queue,
            num_workers: int,
            num_epochs: Optional[int]) -> None:
    """
    Reads Instances from the iterable and puts them in the input_queue.
    """
    logger.info(f"Iterator queuer. PID: {os.getpid()}")
    epoch = 0

    while num_epochs is None or epoch < num_epochs:
        epoch += 1
        for instance in instances:
            input_queue.put(instance)

    # Now put a None for each worker, since each needs to receive one
    # to know that it's done.
    for _ in range(num_workers):
        input_queue.put(None)

@DataIterator.register("multiprocess")
class MultiprocessIterator(DataIterator):
    """
    Wraps another ```DataIterator``` and uses it to generate tensor dicts
    using multiple processes.

    Parameters
    ----------
    base_iterator : ``DataIterator``
        The ``DataIterator`` for generating tensor dicts. It will be shared among
        processes, so it should not be stateful in any way.
    num_workers : ``int``, optional (default = 1)
        The number of processes used for generating tensor dicts.
    output_queue_size: ``int``, optional (default = 1000)
        The size of the output queue on which tensor dicts are placed to be consumed.
        You might need to increase this if you're generating tensor dicts too quickly.
    """
    def __init__(self,
                 base_iterator: DataIterator,
                 num_workers: int = 1,
                 output_queue_size: int = 1000) -> None:
        # pylint: disable=protected-access
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = base_iterator._batch_size
        self.output_queue_size = output_queue_size

        # These two options make the iterator stateful, which means it can't be shared
        # across multiple processes.
        if base_iterator._cache_instances:
            raise ConfigurationError("cannot use Multiprocess iterator with cache_instances")
        if base_iterator._instances_per_epoch:
            raise ConfigurationError("cannot use instances_per_epoch with Multiprocess iterator")

        self.iterator = base_iterator

        self.processes: List[Process] = []
        self.queuer: Optional[Process] = None

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        raise RuntimeError("MultiprocessIterator doesn't use create_batches")

    def index_with(self, vocab: Vocabulary):
        self.iterator.index_with(vocab)

    def _call_with_instances(self,
                             instances: Iterable[Instance],
                             num_epochs: int,
                             shuffle: bool) -> Iterator[TensorDict]:
        # JoinableQueue needed here as sharing tensors across processes
        # requires that the creating process not exit prematurely.
        output_queue = JoinableQueue(self.output_queue_size)
        input_queue = Queue(self.output_queue_size * self.batch_size)

        # Start process that populates the queue.
        self.queuer = Process(target=_queuer,
                              args=(instances, input_queue, self.num_workers, num_epochs))
        self.queuer.start()

        # Start the tensor-dict workers.
        for i in range(self.num_workers):
            args = (input_queue, output_queue, self.iterator, shuffle, i)
            process = Process(target=_create_tensor_dicts_from_queue, args=args)
            process.start()
            self.processes.append(process)

        num_finished = 0
        while num_finished < self.num_workers:
            item = output_queue.get()
            output_queue.task_done()
            if isinstance(item, int):
                num_finished += 1
                logger.info(f"worker {item} finished ({num_finished} / {self.num_workers})")
            else:
                yield item

        for process in self.processes:
            process.join()
        self.processes.clear()

        if self.queuer is not None:
            self.queuer.join()
            self.queuer = None

    def _call_with_qiterable(self,
                             qiterable: QIterable,
                             num_epochs: int,
                             shuffle: bool) -> Iterator[TensorDict]:
        # JoinableQueue needed here as sharing tensors across processes
        # requires that the creating tensor not exit prematurely.
        output_queue = JoinableQueue(self.output_queue_size)

        for _ in range(num_epochs):
            qiterable.start()

            # Start the tensor-dict workers.
            for i in range(self.num_workers):
                args = (qiterable, output_queue, self.iterator, shuffle, i)
                process = Process(target=_create_tensor_dicts_from_qiterable, args=args)
                process.start()
                self.processes.append(process)

            num_finished = 0
            while num_finished < self.num_workers:
                item = output_queue.get()
                output_queue.task_done()
                if isinstance(item, int):
                    num_finished += 1
                    logger.info(f"worker {item} finished ({num_finished} / {self.num_workers})")
                else:
                    yield item

            for process in self.processes:
                process.join()
            self.processes.clear()

            qiterable.join()

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:

        # If you run it forever, the multiprocesses won't shut down correctly.
        # TODO(joelgrus) find a solution for this
        if num_epochs is None:
            raise ConfigurationError("Multiprocess Iterator must be run for a fixed number of epochs")

        if isinstance(instances, QIterable):
            return self._call_with_qiterable(instances, num_epochs, shuffle)
        else:
            return self._call_with_instances(instances, num_epochs, shuffle)

    def __del__(self) -> None:
        """
        Terminate processes if the user hasn't joined implicitly by consuming
        all the tensors. This is necessary as leaving stray processes running
        can corrupt shared state. In brief, we've observed shared memory
        counters being reused (when the memory was free from the perspective of
        the parent process) while the stray workers still held a reference to
        them.

        For a discussion of using destructors in Python in this manner, see
        https://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
        """
        for process in self.processes:
            process.terminate()

        if self.queuer is not None:
            self.queuer.terminate()
