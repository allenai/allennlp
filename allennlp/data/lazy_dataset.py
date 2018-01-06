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
                 instances_per_epoch: int) -> None:
        """
        A LazyDataset just takes a way of generating instances.
        """
        # Call superclass constructor with no instances, we'll override the methods that use them.
        super().__init__([])

        self.generator = generator
        self.iterator = self.generator()
        self.vocab: Vocabulary = None
        self.num_instances = instances_per_epoch

    @overrides
    def truncate(self, max_instances: int):
        raise RuntimeError("cannot truncate a LazyIterator")

    @overrides
    def index_instances(self, vocab: Vocabulary):
        """
        In the ``LazyDataset`` case, we basically use this to grab a reference
        to the ``Vocabulary``.
        """
        self.vocab = vocab


    @overrides
    def __iter__(self) -> Iterator[Instance]:
        if self.vocab is None:
            logger.warning("iterating over lazy dataset that has no vocabulary")
        if self.num_instances is not None:
            try:
                for _ in range(self.num_instances):
                    instance = next(self.iterator)
                    if self.vocab is not None:
                        instance.index_fields(self.vocab)
                    yield instance

            except StopIteration:
                # Got to the end, so reset
                self.iterator = self.generator()
        else:
            yield from self.iterator
            self.iterator = self.generator()

    def __next__(self) -> Instance:
        # First, check if we've reached the end of an epoch,
        # based on the specified instances-per-epoch
        if self.num_instances is not None and self.idx >= self.num_instances:
            self.idx = 0
            raise StopIteration

        try:
            # Get the next instance from ``self.iterator`` and return it.
            self.idx += 1
            instance = next(self.iterator)
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            return instance
        except StopIteration:
            # This error means ``self.iterator`` is finished, so we need to
            # reset the index to 0, grab a fresh iterator, and stop iteration.
            self.idx = 0
            self.iterator = self.generator()
            raise StopIteration

    @overrides
    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        raise NotImplementedError("cannot call get_padding_lengths on a LazyDataset")

    @overrides
    def as_tensor_dict(self,
                       padding_lengths: Dict[str, Dict[str, int]] = None,
                       cuda_device: int = -1,
                       for_training: bool = True,
                       verbose: bool = False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError("cannot call as_tensor_dict on a LazyDataset")
