import logging
from typing import List, Tuple, Dict, Iterable, Generator, Union
from collections import defaultdict

from overrides import overrides
import numpy

from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import BucketIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("epoch_tracking_bucket")
class EpochTrackingBucketIterator(BucketIterator):
    """
    This is essentially a :class:`allennlp.data.iterators.BucketIterator` with just one difference.
    It keeps track of the epoch number, and adds that as an additional meta field to each instance.
    That way, ``Model.forward`` will have access to this information. We do this by keeping track of
    epochs globally, and incrementing them whenever the iterator is called. However, the iterator is
    called both for training and validation sets. So, we keep a dict of epoch numbers, one key per
    dataset.

    Parameters
    ----------
    See :class:`BucketIterator`.
    """
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        super(EpochTrackingBucketIterator, self).__init__(sorting_keys=sorting_keys,
                                                          padding_noise=padding_noise,
                                                          biggest_batch_first=biggest_batch_first,
                                                          batch_size=batch_size,
                                                          instances_per_epoch=instances_per_epoch,
                                                          max_instances_in_memory=max_instances_in_memory)
        # Epoch number value per dataset.
        self._global_epoch_nums: Dict[int, int] = defaultdict(int)

    @overrides
    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1,
                 for_training: bool = True) -> Generator[Dict[str, Union[numpy.ndarray,
                                                                         Dict[str, numpy.ndarray]]],
                                                         None, None]:
        """
        See ``DataIterator.__call__`` for parameters.
        """
        dataset_id = id(instances)
        if num_epochs is None:
            while True:
                self._add_epoch_num_to_instances(instances, dataset_id)
                yield from self._yield_one_epoch(instances, shuffle, cuda_device, for_training)
                self._global_epoch_nums[dataset_id] += 1
        else:
            for _ in range(num_epochs):
                self._add_epoch_num_to_instances(instances, dataset_id)
                yield from self._yield_one_epoch(instances, shuffle, cuda_device, for_training)
                self._global_epoch_nums[dataset_id] += 1

    def _add_epoch_num_to_instances(self,
                                    instances: Iterable[Instance],
                                    dataset_id: int) -> None:
        for instance in instances:
            # TODO(pradeep): Mypy complains here most probably because ``fields`` is typed as a
            # ``Mapping``, and not a ``Dict``. Ignoring this for now, but the type of fields
            # probably needs to be changed.
            instance.fields["epoch_num"] = MetadataField(self._global_epoch_nums[dataset_id])  #type: ignore
