import glob
import logging
import os
from queue import Empty
from typing import List, Iterable, Iterator, Optional

import numpy as np
from torch.multiprocessing import Process, Queue, Value, log_to_stderr

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

        if cls._logger is None:
            cls._logger = log_to_stderr()
            cls._logger.setLevel(logging.INFO)

        cls._logger.info(message)


def _worker(
    reader: DatasetReader,
    input_queue: Queue,
    output_queue: Queue,
    num_active_workers: Value,
    num_inflight_items: Value,
    worker_id: int,
) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.  When
    there are no filenames left on the input queue, it decrements
    num_active_workers to signal completion.
    """
    logger.info(f"Reader worker: {worker_id} PID: {os.getpid()}")
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # It's important that we close and join the queue here before
            # decrementing num_active_workers. Otherwise our parent may join us
            # before the queue's feeder thread has passed all buffered items to
            # the underlying pipe resulting in a deadlock.
            #
            # See:
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#pipes-and-queues
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#programming-guidelines
            output_queue.close()
            output_queue.join_thread()
            # Decrementing is not atomic.
            # See https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Value.
            with num_active_workers.get_lock():
                num_active_workers.value -= 1
            logger.info(f"Reader worker {worker_id} finished")
            break

        logger.info(f"reading instances from {file_path}")
        for instance in reader.read(file_path):
            with num_inflight_items.get_lock():
                num_inflight_items.value += 1
            output_queue.put(instance)


class QIterable(Iterable[Instance]):
    """
    You can't set attributes on Iterators, so this is just a dumb wrapper
    that exposes the output_queue.
    """

    def __init__(self, output_queue_size, epochs_per_read, num_workers, reader, file_path) -> None:
        self.output_queue = Queue(output_queue_size)
        self.epochs_per_read = epochs_per_read
        self.num_workers = num_workers
        self.reader = reader
        self.file_path = file_path

        # Initialized in start.
        self.input_queue: Optional[Queue] = None
        self.processes: List[Process] = []
        # The num_active_workers and num_inflight_items counts in conjunction
        # determine whether there could be any outstanding instances.
        self.num_active_workers: Optional[Value] = None
        self.num_inflight_items: Optional[Value] = None

    def __iter__(self) -> Iterator[Instance]:
        self.start()

        # Keep going as long as not all the workers have finished or there are items in flight.
        while self.num_active_workers.value > 0 or self.num_inflight_items.value > 0:
            # Inner loop to minimize locking on self.num_active_workers.
            while True:
                try:
                    # Non-blocking to handle the empty-queue case.
                    yield self.output_queue.get(block=False, timeout=1.0)
                    with self.num_inflight_items.get_lock():
                        self.num_inflight_items.value -= 1
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
        self.input_queue = Queue(num_shards * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            np.random.shuffle(shards)
            for shard in shards:
                self.input_queue.put(shard)

        # Then put a None per worker to signify no more files.
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        assert (
            not self.processes
        ), "Process list non-empty! You must call QIterable.join() before restarting."
        self.num_active_workers = Value("i", self.num_workers)
        self.num_inflight_items = Value("i", 0)
        for worker_id in range(self.num_workers):
            process = Process(
                target=_worker,
                args=(
                    self.reader,
                    self.input_queue,
                    self.output_queue,
                    self.num_active_workers,
                    self.num_inflight_items,
                    worker_id,
                ),
            )
            logger.info(f"starting worker {worker_id}")
            process.start()
            self.processes.append(process)

    def join(self) -> None:
        for process in self.processes:
            process.join()
        self.processes.clear()

    def __del__(self) -> None:
        """
        Terminate processes if the user hasn't joined. This is necessary as
        leaving stray processes running can corrupt shared state. In brief,
        we've observed shared memory counters being reused (when the memory was
        free from the perspective of the parent process) while the stray
        workers still held a reference to them.

        For a discussion of using destructors in Python in this manner, see
        https://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
        """
        for process in self.processes:
            process.terminate()


@DatasetReader.register("multiprocess")
class MultiprocessDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files using multiple
    processes. Note that in this case the `file_path` passed to `read()` should be a glob, and
    that the dataset reader will return instances from all files matching the glob.  The instances
    will always be read lazily.

    The order the files are processed in is a function of Numpy's random state up to non-determinism
    caused by using multiple worker processes. This can be avoided by setting `num_workers` to 1.

    # Parameters

    base_reader : `DatasetReader`
        Each process will use this dataset reader to read zero or more files.
    num_workers : `int`
        How many data-reading processes to run simultaneously.
    epochs_per_read : `int`, (optional, default=1)
        Normally a call to `DatasetReader.read()` returns a single epoch worth of instances, and
        your `DataIterator` handles iteration over multiple epochs. However, in the
        multiple-process case, it's possible that you'd want finished workers to continue on to the
        next epoch even while others are still finishing the previous epoch. Passing in a value
        larger than 1 allows that to happen.
    output_queue_size : `int`, (optional, default=1000)
        The size of the queue on which read instances are placed to be yielded.
        You might need to increase this if you're generating instances too quickly.
    """

    def __init__(
        self,
        base_reader: DatasetReader,
        num_workers: int,
        epochs_per_read: int = 1,
        output_queue_size: int = 1000,
        **kwargs,
    ) -> None:
        # Multiprocess reader is intrinsically lazy.
        kwargs["lazy"] = True
        super().__init__(**kwargs)

        self.reader = base_reader
        self.num_workers = num_workers
        self.epochs_per_read = epochs_per_read
        self.output_queue_size = output_queue_size

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args, **kwargs)  # type: ignore

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise RuntimeError("Multiprocess reader implements read() directly.")

    def read(self, file_path: str) -> Iterable[Instance]:
        return QIterable(
            output_queue_size=self.output_queue_size,
            epochs_per_read=self.epochs_per_read,
            num_workers=self.num_workers,
            reader=self.reader,
            file_path=file_path,
        )
