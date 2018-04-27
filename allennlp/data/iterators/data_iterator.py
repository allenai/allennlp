import logging
from typing import Dict, Union, Iterable, Iterator, List
from collections import defaultdict
import itertools
import random

import torch

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.utils import add_epoch_number
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name

class DataIterator(Registrable):
    """
    An abstract ``DataIterator`` class. ``DataIterators`` must override ``_create_batches()``.
    """
    default_implementation = 'bucket'
    def __init__(self,
                 cache_instances: bool = False,
                 track_epoch: bool = False):
        self.vocab: Vocabulary = None

        # We might want to cache the instances in memory.
        self._cache_instances = cache_instances
        self._cache: Dict[int, List[TensorDict]] = defaultdict(list)

        # We also might want to add the epoch number to each instance.
        self._track_epoch = track_epoch
        self._epochs: Dict[int, int] = defaultdict(int)

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1,
                 for_training: bool = True) -> Iterator[TensorDict]:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.

        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        cuda_device : ``int``
            If cuda_device >= 0, GPUs are available and Pytorch was compiled with CUDA support, the
            tensor will be copied to the cuda_device specified.
        for_training : ``bool``, optional (default=``True``)
            If ``False``, we will pass the ``volatile=True`` flag when constructing variables,
            which disables gradient computations in the graph.  This makes inference more efficient
            (particularly in memory usage), but is incompatible with training models.
        """
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            self._epochs[key] = epoch

            if self._cache_instances and key in self._cache:
                # Serve the results from the cache.
                tensor_dicts = self._cache[key]

                if shuffle:
                    random.shuffle(tensor_dicts)
                for tensor_dict in tensor_dicts:
                    if self._track_epoch:
                        # The tensor_dict already has an "epoch_num" tensor,
                        # so just fill it with the right value.
                        tensor_dict['epoch_num'].fill_(epoch)
                    yield tensor_dict
            else:
                batches = self._create_batches(instances, shuffle)

                # Should we add the instances to the cache this epoch?
                add_to_cache = self._cache_instances and key not in self._cache

                for batch in batches:
                    if self._track_epoch:
                        add_epoch_number(batch, epoch)

                    if self.vocab is not None:
                        batch.index_instances(self.vocab)

                    padding_lengths = batch.get_padding_lengths()
                    logger.debug("Batch padding lengths: %s", str(padding_lengths))
                    logger.debug("Batch size: %d", len(batch.instances))
                    tensor_dict = batch.as_tensor_dict(padding_lengths,
                                                       cuda_device=cuda_device,
                                                       for_training=for_training)

                    if add_to_cache:
                        self._cache[key].append(tensor_dict)

                    yield tensor_dict

    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        Returns the number of batches that ``dataset`` will be split into; if you want to track
        progress through the batch with the generator produced by ``__call__``, this could be
        useful. If you don't override it, it will always return 1.
        """
        # pylint: disable=unused-argument
        return 1

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DataIterator':
        # TODO(Mark): The adaptive iterator will need a bit of work here,
        # to retrieve the scaling function etc.

        iterator_type = params.pop_choice("type", cls.list_available())
        return cls.by_name(iterator_type).from_params(params)

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab
