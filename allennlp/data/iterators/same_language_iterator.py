from collections import deque, defaultdict
from typing import Iterable, Deque
import logging
import random

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def split_by_language(instance_list):
    insts_by_lang = defaultdict(lambda: [])
    for inst in instance_list:
        inst_lang = inst.fields['metadata'].metadata['lang']
        insts_by_lang[inst_lang].append(inst)

    return iter(insts_by_lang.values())

@DataIterator.register("same_language")
class SameLanguageIterator(BucketIterator):
    """
    Splits batches into batches containing the same language.
    The language of each instance is determined by looking at the 'lang' value
    in the metadata.

    It takes the same parameters as :class:`allennlp.data.iterators.BucketIterator`
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            instance_list = split_by_language(instance_list)
            for same_lang_batch in instance_list:
                iterator = iter(same_lang_batch)
                excess: Deque[Instance] = deque()
                # Then break each memory-sized list into batches.
                for batch_instances in lazy_groups_of(iterator, self._batch_size):
                    for poss_smaller_batches in self._ensure_batch_is_sufficiently_small( # type: ignore
                            batch_instances, excess):
                        batch = Batch(poss_smaller_batches)
                        yield batch
                if excess:
                    yield Batch(excess)
