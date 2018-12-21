from collections import deque
from typing import Iterable, Deque
import logging
import random
import numpy as np
from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DataIterator.register("multiqa")
class BasicIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """


    def __init__(self,
                 all_question_instances_in_batch = False,
                 shuffle: bool = True,
                 batch_size: int = 32) -> None:
        super().__init__(batch_size=batch_size)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._all_question_instances_in_batch = all_question_instances_in_batch

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:

        instances = sorted(instances,key=lambda x: x.fields['metadata'].metadata['question_id'])
        intances_question_id = [instance.fields['metadata'].metadata['question_id'] for instance in instances]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instances[split_inds[ind]:split_inds[ind+1]] for ind in range(len(split_inds)-1)]
        if self._shuffle:
            random.shuffle(per_question_instances)

        batch = []
        for question_instances in per_question_instances:
            # sorting question_instances by rank, we should get them sorted already, but just in case.
            question_instances = sorted(question_instances, key=lambda x: x.fields['metadata'].metadata['rank'])

            if self._all_question_instances_in_batch:
                instances_to_add = question_instances
            else:
                # choose at most 2 instances from the same question:
                if len(question_instances) > 2:
                    # This idea is inspired by Clark and Gardner, 17 - over sample the high ranking documents
                    instances_to_add = random.sample(question_instances[0:2], 1)
                    instances_to_add += random.sample(question_instances[2:], 1)
                else:
                    instances_to_add = question_instances

            # enforcing batch size
            # (for docqa we assume the amount of doucments per question is smaller than batch size)
            if len(batch) + len(instances_to_add) > self._batch_size:
                yield Batch(batch)
                batch = instances_to_add
            else:
                batch += instances_to_add




