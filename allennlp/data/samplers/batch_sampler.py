from typing import List, Iterable, Sequence

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance


class BatchSampler(Registrable):
    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        raise NotImplementedError

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        raise NotImplementedError
