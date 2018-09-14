from typing import List, Iterable, Iterator
import glob
import logging
import random

from torch.multiprocessing import Manager, Process, Queue, log_to_stderr

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

logger = log_to_stderr()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

def _worker(reader: DatasetReader,
            input_queue: Queue,
            output_queue: Queue,
            index: int) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.
    When there are no filenames left on the input queue, it puts its ``index``
    on the output queue and doesn't do anything else.
    """
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # Put my index on the queue to signify that I'm finished
            output_queue.put(index)
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
                self.num_workers = outer_self.num_workers

            def __iter__(self) -> Iterator[Instance]:
                # pylint: disable=protected-access
                return outer_self._instances(file_path, self.manager, self.output_queue)

        return QIterable()

    def _instances(self, file_path: str, manager: Manager, output_queue: Queue) -> Iterator[Instance]:
        """
        A generator that reads instances off the output queue and yields them up
        until none are left (signified by all ``num_workers`` workers putting their
        ids into the queue).
        """
        shards = glob.glob(file_path)
        num_shards = len(shards)

        # If we want multiple epochs per read, put shards in the queue multiple times.
        input_queue = manager.Queue(num_shards * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            random.shuffle(shards)
            for shard in shards:
                input_queue.put(shard)

        # Then put a None per worker to signify no more files.
        for _ in range(self.num_workers):
            input_queue.put(None)

        processes: List[Process] = []
        num_finished = 0

        for worker_id in range(self.num_workers):
            process = Process(target=_worker,
                              args=(self.reader, input_queue, output_queue, worker_id))
            logger.info(f"starting worker {worker_id}")
            process.start()
            processes.append(process)

        # Keep going as long as not all the workers have finished.
        while num_finished < self.num_workers:
            item = output_queue.get()
            if isinstance(item, int):
                # Means a worker has finished, so increment the finished count.
                num_finished += 1
                logger.info(f"worker {item} finished ({num_finished}/{self.num_workers})")
            else:
                # Otherwise it's an ``Instance``, so yield it up.
                yield item

        for process in processes:
            process.join()
        processes.clear()
