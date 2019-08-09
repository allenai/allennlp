from queue import Empty
from typing import List, Iterable, Iterator
import glob
import logging
import os

import numpy as np
from torch import multiprocessing
from torch.multiprocessing import Manager, Process, Queue, Value, log_to_stderr

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

class logger:
    """
    multiprocessing.log_to_stderr causes some output in the logs
    even when we don't use this dataset reader. This is a small hack
    to instantiate the stderr logger lazily only when it's needed
    (which is only when using the MultiprocessDatasetReader)
    """
    _logger = None

    @classmethod
    def info(cls, message: str) -> None:
        # pylint: disable=no-self-use
        if cls._logger is None:
            cls._logger = log_to_stderr()
            cls._logger.setLevel(logging.INFO)

        cls._logger.info(message)


def _worker(reader: DatasetReader,
            input_queue: Queue,
            output_queue: Queue,
            active_workers: Value,
            inflight_items: Value,
            worker_id: int) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.  When
    there are no filenames left on the input queue, it decrements
    active_workers to signal completion.
    """
    logger.info(f"Reader worker: {worker_id} PID: {os.getpid()}")
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # It's important that we close and join the queue here before
            # decrementing active_workers. Otherwise our parent may join us
            # before the queue's feeder thread has passed all buffered items to
            # the underlying pipe resulting in a deadlock.
            #
            # See:
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#pipes-and-queues
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#programming-guidelines
            output_queue.close()
            output_queue.join_thread()
            # Decrementing is not atomic. See https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Value.
            with active_workers.get_lock():
                active_workers.value -= 1
            logger.info(f"Reader worker {worker_id} finished")
            break

        logger.info(f"reading instances from {file_path}")
        for instance in reader.read(file_path):
            with inflight_items.get_lock():
                inflight_items += 1
            output_queue.put(instance)


class QIterable(Iterable[Instance]):
    """
    You can't set attributes on Iterators, so this is just a dumb wrapper
    that exposes the output_queue.
    """
    def __init__(self, output_queue_size, epochs_per_read, num_workers, reader, file_path) -> None:
        #self.manager = Manager()
        #self.output_queue = self.manager.Queue(output_queue_size)
        ctx = multiprocessing.get_context("fork")
        self.output_queue = ctx.Queue(output_queue_size)
        self.epochs_per_read = epochs_per_read
        self.num_workers = num_workers
        self.reader = reader
        self.file_path = file_path

    def __iter__(self) -> Iterator[Instance]:
        self.start()

        # Keep going as long as not all the workers have finished or there are items in flight.
        while self.active_workers.value > 0 or self.inflight_items > 0:
            # Inner loop to minimize locking on self.active_workers.
            while True:
                try:
                    # Non-blocking to handle the empty-queue case.
                    yield self.output_queue.get(block=False, timeout=1.0)
                    with self.inflight_items.get_lock():
                        self.inflight_items -= 1
                except Empty:
                    # The queue could be empty because the workers are
                    # all finished or because they're busy processing.
                    # The outer loop distinguishes between these two
                    # cases.
                    break

        self.join()

    def start(self) -> None:
        shards = glob.glob(self.file_path)
        # Ensure a consistent order before shuffling for testing.
        shards.sort()
        num_shards = len(shards)

        # If we want multiple epochs per read, put shards in the queue multiple times.
        #self.input_queue = self.manager.Queue(num_shards * self.epochs_per_read + self.num_workers)
        ctx = multiprocessing.get_context("fork")
        self.input_queue = ctx.Queue(num_shards * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            np.random.shuffle(shards)
            for shard in shards:
                self.input_queue.put(shard)

        # Then put a None per worker to signify no more files.
        for _ in range(self.num_workers):
            self.input_queue.put(None)


        self.processes: List[Process] = []
        # active_workers and inflight_items in conjunction determine whether there could be any outstanding instances.
        self.active_workers = ctx.Value('i', self.num_workers)
        self.inflight_items = ctx.Value('i', 0)
        ctx = multiprocessing.get_context("fork")
        for worker_id in range(self.num_workers):
            process = ctx.Process(target=_worker,
                                  args=(self.reader, self.input_queue, self.output_queue,
                                        self.active_workers, self.inflight_items, worker_id),
                                  daemon=True)
            logger.info(f"starting worker {worker_id}")
            process.start()
            self.processes.append(process)

    def join(self) -> None:
        for i, process in enumerate(self.processes):
            process.join()
            # Best effort logging. It's entirely possible it finished earlier.
            logger.info(f"worker: {i} alive: {process.is_alive()}")
        self.processes.clear()

@DatasetReader.register('multiprocess')
class MultiprocessDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files
    using multiple processes. Note that in this case the ``file_path`` passed to ``read()``
    should be a glob, and that the dataset reader will return instances from all files
    matching the glob.

    The order the files are processed in is a function of Numpy's random state
    up to non-determinism caused by using multiple worker processes. This can
    be avoided by setting ``num_workers`` to 1.

    Parameters
    ----------
    base_reader : ``DatasetReader``
        Each process will use this dataset reader to read zero or more files.
    num_workers : ``int``
        How many data-reading processes to run simultaneously.
    epochs_per_read : ``int``, (optional, default=1)
        Normally a call to ``DatasetReader.read()`` returns a single epoch worth of instances,
        and your ``DataIterator`` handles iteration over multiple epochs. However, in the
        multiple-process case, it's possible that you'd want finished workers to continue on to the
        next epoch even while others are still finishing the previous epoch. Passing in a value
        larger than 1 allows that to happen.
    output_queue_size: ``int``, (optional, default=1000)
        The size of the queue on which read instances are placed to be yielded.
        You might need to increase this if you're generating instances too quickly.
    """
    def __init__(self,
                 base_reader: DatasetReader,
                 num_workers: int,
                 epochs_per_read: int = 1,
                 output_queue_size: int = 1000) -> None:
        # Multiprocess reader is intrinsically lazy.
        super().__init__(lazy=True)

        self.reader = base_reader
        self.num_workers = num_workers
        self.epochs_per_read = epochs_per_read
        self.output_queue_size = output_queue_size

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        # pylint: disable=arguments-differ
        return self.reader.text_to_instance(*args, **kwargs)

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise RuntimeError("Multiprocess reader implements read() directly.")

    def read(self, file_path: str) -> Iterable[Instance]:
        return QIterable(
                output_queue_size=self.output_queue_size,
                epochs_per_read=self.epochs_per_read,
                num_workers=self.num_workers,
                reader=self.reader,
                file_path=file_path
        )
