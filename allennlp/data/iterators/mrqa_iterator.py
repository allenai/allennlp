import logging
from typing import List, Tuple, Iterable

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DataIterator.register("mrqa_iterator")
class MRQAIterator(DataIterator):
    """
    This iterator groups instances by question_id and creates batches so that all instances
    of the same question_id are in the same batch.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """
    def __init__(self,
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 one_instance_per_batch = False,
                 maximum_tensor_size: int = None,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._maximum_tensor_size = maximum_tensor_size
        self._one_instance_per_batch = one_instance_per_batch

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            # organizing instances per question
            intances_question_id = [instance.fields['metadata'].metadata['question_id'] for instance in instance_list]
            split_inds = [0]
            for ind in range(len(intances_question_id)-1):
                if intances_question_id[ind] != intances_question_id[ind+1]:
                    split_inds.append(ind + 1)
            split_inds += [len(intances_question_id)]
            per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in range(len(split_inds) - 1)]

            batch = []
            for question_instances in per_question_instances:
                instances_to_add = question_instances
                if (len(batch) + len(instances_to_add) > self._batch_size and len(batch) > 0):
                    batch = sorted(batch,key=lambda x: x.fields['metadata'].metadata['question_id'])
                    yield Batch(batch)

                    batch = instances_to_add
                else:
                    batch += instances_to_add

            # yielding remainder batch
            if len(batch)>0:
                batch = sorted(batch, key=lambda x: x.fields['metadata'].metadata['question_id'])
                yield Batch(batch)




