from typing import List, Iterable, Iterator
import glob
import logging
from queue import Empty

import numpy as np
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
            active_workers: Value) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.  When
    there are no filenames left on the input queue, it decrements
    active_workers to signal completion.
    """
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # Decrementing is not atomic. See https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Value.
            with active_workers.get_lock():
                active_workers.value -= 1
            break

        logger.info(f"reading instances from {file_path}")
        for instance in reader.read(file_path):
            output_queue.put(instance)



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
        outer_self = self

        class QIterable(Iterable[Instance]):
            """
            You can't set attributes on Iterators, so this is just a dumb wrapper
            that exposes the output_queue. Currently you probably shouldn't touch
            the output queue, but this is done with an eye toward implementing
            a data iterator that can read directly from the queue (instead of having
            to use the _instances iterator we define here.)
            """
            def __init__(self) -> None:
                self.manager = Manager()
                self.output_queue = self.manager.Queue(outer_self.output_queue_size)

            def __iter__(self) -> Iterator[Instance]:
                self.start()

                # Keep going as long as not all the workers have finished.
                while self.active_workers.value > 0:
                    # Inner loop to minimize locking on self.active_workers.
                    while True:
                        try:
                            # Non-blocking to handle the empty-queue case.
                            yield self.output_queue.get(block=False, timeout=1.0)
                        except Empty:
                            # The queue could be empty because the workers are
                            # all finished or because they're busy processing.
                            # The outer loop distinguishes between these two
                            # cases.
                            break

                self.join()

            def start(self) -> None:
                shards = glob.glob(file_path)
                # Ensure a consistent order before shuffling for testing.
                shards.sort()
                num_shards = len(shards)

                # If we want multiple epochs per read, put shards in the queue multiple times.
                self.input_queue = self.manager.Queue(num_shards * outer_self.epochs_per_read + outer_self.num_workers)
                for _ in range(outer_self.epochs_per_read):
                    np.random.shuffle(shards)
                    for shard in shards:
                        self.input_queue.put(shard)

                # Then put a None per worker to signify no more files.
                for _ in range(outer_self.num_workers):
                    self.input_queue.put(None)


                self.processes: List[Process] = []
                self.active_workers = Value('i', outer_self.num_workers)
                for worker_id in range(outer_self.num_workers):
                    process = Process(target=_worker,
                                      args=(outer_self.reader, self.input_queue, self.output_queue, self.active_workers))
                    logger.info(f"starting worker {worker_id}")
                    process.start()
                    self.processes.append(process)

            def join(self) -> None:
                for i, process in enumerate(self.processes):
                    process.join()
                    # Best effort logging. It's entirely possible it finished earlier.
                    logger.info(f"worker {i} finished")
                self.processes.clear()

        return QIterable()
