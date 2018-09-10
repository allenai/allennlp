from typing import List, Iterable
import glob
import logging
import queue
import random

from torch.multiprocessing import Process, Queue

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name


def _worker(reader: DatasetReader,
            input_queue: Queue,
            output_queue: Queue) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.
    When there are no filenames left on the input queue, it puts ``None``
    on the output queue and doesn't do anything else.
    """
    # Keep going as long as there are things on the queue
    while True:
        try:
            # All of the possible files are put on the queue before
            # any workers are even started, so if the queue is empty
            # it really means there's no more work to do. Which means
            # that ``get_nowait`` and breaking out of the loop are correct.
            file_path = input_queue.get_nowait()
        except queue.Empty:
            break

        for instance in reader.read(file_path):
            output_queue.put(instance)

    # Put None on the queue to signify that we've finished
    output_queue.put(None)


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
    lazy : ``bool``, (optional, default=True)
        Most of our dataset readers are eager by default; however, if you're using this one
        it's probably because you have a lot of data (and in particular don't want to load it
        all into memory), so laziness is the default.
    """
    def __init__(self,
                 base_reader: DatasetReader,
                 num_workers: int,
                 epochs_per_read: int = 1,
                 lazy: bool = True) -> None:
        super().__init__(lazy=lazy)

        self.reader = base_reader
        self.num_workers = num_workers
        self.epochs_per_read = epochs_per_read

    def text_to_instance(self, *args, **kwargs) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        # pylint: disable=arguments-differ
        return self.reader.text_to_instance(*args, **kwargs)

    def _read(self, file_path: str) -> Iterable[Instance]:
        shards = glob.glob(file_path)
        num_shards = len(shards)

        # If we want multiple epochs per read, put shards in the queue multiple times.
        input_queue = Queue(num_shards * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            random.shuffle(shards)
            for shard in shards:
                input_queue.put(shard)

        # TODO(joelgrus): where does this number come from?
        output_queue = Queue(1000)

        processes: List[Process] = []
        num_finished = 0

        for worker_id in range(self.num_workers):
            process = Process(target=_worker,
                              args=(self.reader, input_queue, output_queue))
            logger.info(f"starting worker {worker_id}")
            process.start()
            processes.append(process)

        # Keep going as long as not all the workers have finished.
        while num_finished < self.num_workers:
            item = output_queue.get()
            if item is None:
                # None means a worker has finished, so increment the finished count.
                num_finished += 1
                logger.info(f"{num_finished}/{self.num_workers} finished")
            else:
                # Otherwise it's an ``Instance``, so yield it up.
                yield item

        # Once we know all the workers are done, join all the processes.
        logger.info("done reading, joining processes")
        for process in processes:
            process.join()
        processes.clear()
