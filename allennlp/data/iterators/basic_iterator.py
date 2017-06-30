from typing import List

from ..instance import Instance
from ..dataset import Dataset
from ...common.util import group_by_count


class BasicIterator:
    """
    A very basic iterator, which takes a dataset, pads all of it's instances to
    the maximum lengths of the relevant fields across the whole dataset and yields
    fixed size batches.

    Parameters
    ----------
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    """
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def __call__(self, dataset: Dataset):
        grouped_instances = self._create_batches(dataset)
        self.last_num_batches = len(grouped_instances)  # pylint: disable=attribute-defined-outside-init

        padding_lengths = dataset.get_padding_lengths()
        while True:
            for group in grouped_instances:
                batch = Dataset(group)
                yield batch.as_arrays(padding_lengths, verbose=False)

    def _create_batches(self, dataset: Dataset) -> List[List[Instance]]:
        instances = dataset.instances
        grouped_instances = group_by_count(instances, self.batch_size, None)
        # The last group might have not been full, so we check if any of the instances
        # are None, which is how group_by_count pads non-complete batches.
        grouped_instances[-1] = [instance for instance in grouped_instances[-1] if instance is not None]
        return grouped_instances
