from collections import deque
from typing import Iterable, Deque
import logging
import random
import numpy as np
from overrides import overrides

import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

@profile
def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        # We use instance[0] - the highest ranking document within the question instances, to sort.
        instance[0].index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance[0].get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

def calc_tensor_size(instances: List[Instance],
        sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
        vocab: Vocabulary):
    estimated_tensor_size = 0
    for instance in instances:
        instance_tensor_size = 1
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        for (field_name, padding_key) in sorting_keys:
            instance_tensor_size *= padding_lengths[field_name][padding_key]
        if instance_tensor_size > estimated_tensor_size:
            estimated_tensor_size =  instance_tensor_size

    return estimated_tensor_size * len(instances) 
    

@DataIterator.register("multiqa")
class MultiQAIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 all_question_instances_in_batch = False,
                 one_instance_per_batch = False,
                 maximum_tensor_size: int = None,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._maximum_tensor_size = maximum_tensor_size
        self._one_instance_per_batch = one_instance_per_batch
        self._all_question_instances_in_batch = all_question_instances_in_batch

    @profile
    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        temp_max_tensor_size = 0
        #all_instances_ids = []
        total_examples_in_batches = 0
        for instance_list in self._memory_sized_lists(instances):
            #all_instances_ids += [inst.fields['metadata'].metadata['instance_id'] for inst in instance_list]
            #logger.info("itertor: total instances %d, unique instances %d ", len(all_instances_ids), len(set(all_instances_ids)))
            if self._one_instance_per_batch:
                instances_to_add = []
                for instance in instance_list:
                    instances_to_add += [instance]
                    if len(instances_to_add) >= self._batch_size:
                        yield Batch(instances_to_add)
                        instances_to_add = []

                # remainder
                if len(instances_to_add) > 0:
                    yield Batch(instances_to_add)
            else:
                # if empty sorting_keys - we assume no sort is required
                if self._sorting_keys != [[]]:
                    instance_list = sorted(instance_list,key=lambda x: x.fields['metadata'].metadata['question_id'])
                    intances_question_id = [instance.fields['metadata'].metadata['question_id'] for instance in instance_list]
                    split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
                    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind+1]] for ind in range(len(split_inds)-1)]

                    # sorting question_instances by rank, we should get them sorted already, but just in case.
                    for ind in range(len(per_question_instances)):
                        per_question_instances[ind] = sorted(per_question_instances[ind], key=lambda x: x.fields['metadata'].metadata['rank'])

                    per_question_instances = sort_by_padding(per_question_instances,
                                                             self._sorting_keys,
                                                             self.vocab,
                                                             self._padding_noise)
                else:
                    intances_question_id = [instance.fields['metadata'].metadata['question_id'] for instance in instance_list]
                    split_inds = [0]
                    for ind in range(len(intances_question_id)-1):
                        if intances_question_id[ind] != intances_question_id[ind+1]:
                            split_inds.append(ind + 1)
                    split_inds += [len(intances_question_id)]
                    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in range(len(split_inds) - 1)]


                batch = []
                for question_instances in per_question_instances:

                    if self._all_question_instances_in_batch:
                        instances_to_add = question_instances
                    else:
                        # choose at most 2 instances from the same question:
                        if len(question_instances) > 2:
                            # This part is inspired by Clark and Gardner, 17 - oversample the highest ranking documents.
                            # In thier work they use only instances with answers, so we will find the highest
                            # ranking instance with an answer (this also insures we have at least one answer in the chosen instances)
                            inst_with_answers = [inst for inst in question_instances if inst.fields['metadata'].metadata['has_answer']]

                            # Because we cannot enforce all instances of the same question to be
                            # provided in the same _memory_sized_lists we have to check this:
                            # yet another reason to do this in the data reader
                            if len(inst_with_answers) > 0:
                                instances_to_add = random.sample(inst_with_answers[0:2], 1)
                                # we assume each question will be visited once in an epoch
                                question_instances.remove(instances_to_add[0])
                            else:
                                instances_to_add = []
                            instances_to_add += random.sample(question_instances, 1)

                        else:
                            instances_to_add = question_instances

                        # Require at least one answer:
                        if not any(inst.fields['metadata'].metadata['answers_list'] != [] for inst in instances_to_add):
                            continue

                    # enforcing batch size
                    # (for docqa we assume the amount of doucments per question is smaller than batch size)
                    if self._maximum_tensor_size is not None:
                        estimated_tensor_size = calc_tensor_size(batch + instances_to_add,
                             self._sorting_keys,
                             self.vocab)
                        if estimated_tensor_size > temp_max_tensor_size:
                            temp_max_tensor_size = estimated_tensor_size
                            #logger.info("temp_max_tensor_size = %d",temp_max_tensor_size)

                    if (len(batch) + len(instances_to_add) > self._batch_size and len(batch) > 0) or \
                        (self._maximum_tensor_size is not None and estimated_tensor_size > self._maximum_tensor_size):
                        #total_examples_in_batches += len(batch)
                        #logger.info("total_examples_in_batches = %d", total_examples_in_batches)

                        # This is done for a bug in inside the model in dev set .... i need to get rid of this and refactor everything here
                        batch = sorted(batch,key=lambda x: x.fields['metadata'].metadata['question_id'])
                        yield Batch(batch)

                        batch = instances_to_add
                    else:
                        batch += instances_to_add

                # yielding remainder batch
                if len(batch)>0:
                    #total_examples_in_batches += len(batch)
                    #logger.info("remainder ... total_examples_in_batches = %d", total_examples_in_batches)

                    # This is done for a bug in inside the model in dev set .... i need to get rid of this and refactor everything here
                    batch = sorted(batch, key=lambda x: x.fields['metadata'].metadata['question_id'])
                    yield Batch(batch)




