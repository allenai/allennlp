from typing import List, Tuple
from copy import deepcopy
from overrides import overrides
from ..instance import Instance
from ..dataset import Dataset
from .basic_iterator import BasicIterator
from ...common.util import add_noise_to_dict_values


class BucketIterator(BasicIterator):

    """
    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]], optional (default = [])
        To bucket inputs into batches, we want to group the instances by padding length, so that
        we minimize the amount of padding necessary per batch. By default, this sorting
    padding_noise: double, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    sort_every_epoch: bool, optional (default=True)
        If ``True``, we will re-sort the data after every epoch, then re-group the instances into
        batches.  If ``padding_noise`` is zero, this does nothing, but if it's non-zero, this will
        give you a slightly different ordering, so you don't have exactly the same batches at every
        epoch.  If you're doing adaptive batch sizes, this will lead to re-computing the adaptive
        batches each epoch, which could give a different number of batches for the whole dataset,
        which means each "epoch" might no longer correspond to `exactly` one pass over the data.
        This is probably a pretty minor issue, though.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]] = None,
                 padding_noise: float = 0.2,
                 sort_every_epoch: bool = True,
                 batch_size: int = 32):

        self.sorting_keys = sorting_keys or []
        self.padding_noise = padding_noise
        self.sort_every_epoch = sort_every_epoch
        super(BucketIterator, self).__init__(batch_size)

    @overrides
    def __call__(self, dataset: Dataset):
        grouped_instances = self._create_batches(dataset)
        self.last_num_batches = len(grouped_instances)

        while True:
            if self.sort_every_epoch:
                unpadded_dataset = deepcopy(dataset)
                groups = self._create_batches(unpadded_dataset)
            else:
                groups = grouped_instances
            for group in groups:
                batch = Dataset(group)
                batch_padding_lengths = batch.get_padding_lengths()
                yield batch.as_arrays(batch_padding_lengths, verbose=False)

    @overrides
    def _create_batches(self, dataset: Dataset) -> List[List[Instance]]:
        if self.sorting_keys:
            dataset = self.sort_dataset_by_padding(dataset,
                                                   self.sorting_keys,
                                                   self.padding_noise)
        return super(BucketIterator, self)._create_batches(dataset)

    @staticmethod
    def sort_dataset_by_padding(dataset: Dataset,
                                sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                                padding_noise: float = 0.0) -> Dataset:
        """
        Sorts the ``Instances`` in this ``Dataset`` by their padding lengths, using the keys in
        ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
        ``(field_name, padding_key)`` tuples.
        """
        instances_with_lengths = []
        for instance in dataset.instances:
            padding_lengths = instance.get_padding_lengths()
            if padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths:
                    noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
                padding_lengths = noisy_lengths
            instance_with_lengths = [padding_lengths[field_name][padding_key]
                                     for (field_name, padding_key) in sorting_keys] + [instance]
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[:-1])
        return Dataset([instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths])
