import logging
from typing import Dict, Generator, Union, Iterable

import numpy

from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DataIterator(Registrable):
    """
    An abstract ``DataIterator`` class. ``DataIterators`` must implement __call__, which yields
    batched examples.
    """
    default_implementation = 'bucket'
    vocab: Vocabulary = None

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True,
                 cuda_device: int = -1,
                 for_training: bool = True) -> Generator[Dict[str, Union[numpy.ndarray,
                                                                         Dict[str, numpy.ndarray]]],
                                                         None, None]:
        """
        Returns a generator that yields batches over the given dataset, forever.

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
        if num_epochs is None:
            while True:
                yield from self._yield_one_epoch(instances, shuffle, cuda_device, for_training)
        else:
            for _ in range(num_epochs):
                yield from self._yield_one_epoch(instances, shuffle, cuda_device, for_training)

    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        Returns the number of batches that ``dataset`` will be split into; if you want to track
        progress through the batch with the generator produced by ``__call__``, this could be
        useful.
        """
        raise NotImplementedError

    def _yield_one_epoch(self, instances: Iterable[Instance], shuffle: bool, cuda_device: int, for_training: bool):
        batches = self._create_batches(instances, shuffle)
        for batch in batches:
            if self.vocab is not None:
                batch.index_instances(self.vocab)
            padding_lengths = batch.get_padding_lengths()
            logger.debug("Batch padding lengths: %s", str(padding_lengths))
            logger.debug("Batch size: %d", len(batch.instances))
            yield batch.as_tensor_dict(padding_lengths,
                                       cuda_device=cuda_device,
                                       for_training=for_training)

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        Creates batches of instances. Each batch is a small ``Dataset``.
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
