from typing import Iterable, Iterator, List, Optional
import logging

from torch.multiprocessing import Process, Queue

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.dataset import Batch
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _create_tensor_dicts(input_queue: Queue,
                         output_queue: Queue,
                         max_instances_in_memory: int,
                         batch_size: int,
                         cuda_device: int,
                         vocab: Vocabulary) -> None:
    """
    Pulls at most ``max_instances_in_memory`` from the input_queue,
    groups them into batches of size ``batch_size``, converts them
    to ``TensorDict`` s, and puts them on the ``output_queue``.
    """
    instances: List[Instance] = []

    def make_batches() -> None:
        for batch_instances in lazy_groups_of(iter(instances), batch_size):
            batch = Batch(batch_instances)

            if vocab is not None:
                batch.index_instances(vocab)

            padding_lengths = batch.get_padding_lengths()
            tensor_dict = batch.as_tensor_dict(padding_lengths,
                                               cuda_device=cuda_device)

            output_queue.put(tensor_dict)
        instances.clear()

    while True:
        instance = input_queue.get()
        if instance is None:
            # no more instances, create one last batch, and stick None there.
            if instances:
                make_batches()
            output_queue.put(None)
            break
        else:
            instances.append(instance)
            if len(instances) >= max_instances_in_memory:
                make_batches()

def _queuer(instances: Iterable[Instance],
            input_queue: Queue,
            num_workers: int,
            num_epochs: Optional[int]) -> None:
    """
    Reads Instances from the iterable and puts them in the input_queue.
    """
    epoch = 0

    while num_epochs is None or epoch < num_epochs:
        epoch += 1
        for instance in instances:
            input_queue.put(instance)

    # now put a None for each worker
    for _ in range(num_workers):
        input_queue.put(None)

@DataIterator.register("multiprocess")
class MultiprocessIterator(DataIterator):
    def __init__(self,
                 num_workers: int = 1,
                 max_instances_in_memory: int = 1000,
                 batch_size: int = 32) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_instances_in_memory = max_instances_in_memory
        self.vocab: Vocabulary = None

        self.processes: List[Process] = []
        self.queuer: Optional[Process] = None

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        raise RuntimeError("MultiprocessIterator doesn't use create_batches")

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1) -> Iterator[TensorDict]:

        # If you run it forever, the multiprocesses won't shut down correctly.
        # TODO(joelgrus) find a solution for this
        if num_epochs is None:
            raise ConfigurationError("Multiprocess Iterator must be run for a fixed number of epochs")

        # TODO(joelgrus) are these the right sizes?
        input_queue = Queue(1000)
        output_queue = Queue(1000)

        # Start the tensor-dict workers.
        for _ in range(self.num_workers):
            args = (input_queue, output_queue,
                    self.max_instances_in_memory, self.batch_size, cuda_device, self.vocab)

            process = Process(target=_create_tensor_dicts, args=args)
            process.start()
            self.processes.append(process)

        # Start the queue-populating worker.
        self.queuer = Process(target=_queuer, args=(instances, input_queue, self.num_workers, num_epochs))
        self.queuer.start()

        num_finished = 0
        while num_finished < self.num_workers:
            item = output_queue.get()
            if item is None:
                # finished
                num_finished += 1
                print(num_finished, self.num_workers)
            else:
                yield item

        self.queuer.join()
        self.queuer = None

        for process in self.processes:
            process.join()
        self.processes.clear()
