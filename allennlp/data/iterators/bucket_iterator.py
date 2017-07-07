from typing import List, Tuple
import random

from overrides import overrides

from allennlp.common.util import add_noise_to_dict_values
from allennlp.data import Dataset, Instance
from allennlp.data.iterators.basic_iterator import BasicIterator


class BucketIterator(BasicIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]], optional (default = [])
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        For example:
        ``[("sentence1", "sentence_length"),
           ("sentence2", "sentence_length"),
           ("sentence1", "char_length")]``

        would sort a dataset first by the "sentence_length" of the "sentence1" field, then by the
        "sentence_length" of the "sentence2" field, and finally by the "char_length" of the
        "sentence1" field.

        By default, the list of sorting keys is empty, meaning the dataset won't be sorted and
        batches will just be padded using the max lengths of all fields requiring padding
        calculated per batch.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]] = None,
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32):
        self._sorting_keys = sorting_keys or []
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        super(BucketIterator, self).__init__(batch_size)

    @overrides
    def _create_batches(self, dataset: Dataset, shuffle: bool = True) -> List[List[Instance]]:
        if self._sorting_keys:
            dataset = self._sort_dataset_by_padding(dataset,
                                                    self._sorting_keys,
                                                    self._padding_noise)
        grouped_instances = super(BucketIterator, self)._create_batches(dataset, shuffle=False)
        if self._biggest_batch_first:
            # We'll actually pop the last _two_ batches, because the last one might not be full.
            last_batch = grouped_instances.pop()
            penultimate_batch = grouped_instances.pop()
        if shuffle:
            random.shuffle(grouped_instances)
        if self._biggest_batch_first:
            grouped_instances.insert(0, penultimate_batch)
            grouped_instances.insert(0, last_batch)
        return super(BucketIterator, self)._create_batches(dataset)

    @staticmethod
    def _sort_dataset_by_padding(dataset: Dataset,
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
