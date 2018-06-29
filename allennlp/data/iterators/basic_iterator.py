from typing import Iterable
import logging
import random

from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("basic")
class BasicIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
#                if self._maximum_samples_per_batch:
#                    # check if we need to break into smaller chunks
#                    key, limit = self._maximum_samples_per_batch
#                    padding_length = -1
#                    list_batch_instances = list(batch_instances)
#                    for instance in list_batch_instances:
#                        field_lengths = instance.get_padding_lengths()
#                        for _, lengths in field_lengths.items():
#                            try:
#                                padding_length = max(padding_length,
#                                                     lengths[key])
#                            except KeyError:
#                                pass
#
#                    if padding_length * len(list_batch_instances) > limit:
#                        # need to shrink
#                        num_samples = padding_length * len(list_batch_instances)
#                        num_shrunk_batches = math.ceil(num_samples / float(limit))
#                        shrunk_batch_size = math.ceil(len(list_batch_instances) / num_shrunk_batches)
#                        start = 0
#                        while start < len(list_batch_instances):
#                            end = start + shrunk_batch_size
#                            yield Batch(list_batch_instances[start:end])
#                            start = end
#                    else:
#                        yield Batch(batch_instances)
#                else:
#                    yield Batch(batch_instances)
                batch = Batch(batch_instances)
                yield batch

    @classmethod
    def from_params(cls, params: Params) -> 'BasicIterator':
        batch_size = params.pop_int('batch_size', 32)
        instances_per_epoch = params.pop_int('instances_per_epoch', None)
        max_instances_in_memory = params.pop_int('max_instances_in_memory', None)
        maximum_samples_per_batch = params.pop("maximum_samples_per_batch", None)
        cache_instances = params.pop_bool('cache_instances', False)
        track_epoch = params.pop_bool('track_epoch', False)

        params.assert_empty(cls.__name__)
        return cls(batch_size=batch_size,
                   instances_per_epoch=instances_per_epoch,
                   max_instances_in_memory=max_instances_in_memory,
                   maximum_samples_per_batch=maximum_samples_per_batch)
                   cache_instances=cache_instances,
                   track_epoch=track_epoch)
