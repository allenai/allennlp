"""
A :class:`~LazyDataset` is a subclass of :class:`~Dataset` that
instead of storing a ``List`` of ``Instance`` s, stores a method
for generating instances. The main intended use case is for datasets
that are too large to load into memory, in which case the generator
would just read the file from disk for each call to ``iterinstances``.
"""
import logging
from typing import Dict, Union, Callable, Iterator

from overrides import overrides
import torch

from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LazyDataset(Dataset):
    """
    A lazy collection of :class:`~allennlp.data.instance.Instance` objects.
    The ``Instances`` have ``Fields``, and the fields
    could be in an indexed or unindexed state - the ``Dataset`` has methods around indexing the
    data and converting the data into arrays.
    """
    def __init__(self,
                 generator: Callable[[], Iterator[Instance]],
                 num_instances_per_epoch: int = None) -> None:
        """
        A LazyDataset just takes a way of generating instances.
        Each call to ``generator()`` should return an Iterator
        representing one epoch worth of Instances (which may be
        your entire dataset and may not be).
        """
        super().__init__(num_instances_per_epoch)

        self.generator = generator
        self.vocab: Vocabulary = None

    @overrides
    def index_instances(self, vocab: Vocabulary):
        """
        In the ``LazyDataset`` case, we basically use this to grab a reference
        to the ``Vocabulary``. We'll index instances as we generate them.
        """
        self.vocab = vocab

    def _indexed(self, instance: Instance) -> Instance:
        if self.vocab is not None:
            instance.index_fields(self.vocab)
        return instance

    @overrides
    def __iter__(self) -> Iterator[Instance]:
        if self.vocab is None:
            logger.warning("iterating over lazy dataset that has no vocabulary")

        for instance in self.generator():
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance
