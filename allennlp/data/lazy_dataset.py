"""
A :class:`~LazyDataset` is essentially a :class:`~Dataset` that lives on disk
instead of in-memory. It's not a subclass of ``Dataset`` because it doesn't
behave quite the same.
"""
import logging
from collections import defaultdict
from typing import Dict, Union, Callable, Iterator

import torch
import tqdm

from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LazyDataset(Dataset):
    """
    A collection of :class:`~allennlp.data.instance.Instance` objects.
    The ``Instances`` have ``Fields``, and the fields
    could be in an indexed or unindexed state - the ``Dataset`` has methods around indexing the
    data and converting the data into arrays.
    """
    def __init__(self, generator: Callable[[], Iterator[Instance]]) -> None:
        """
        A LazyDataset just takes a way of generating instances.
        """
        # Call superclass constructor with no instances, we'll override the methods that use them.
        super().__init__([])

        self.generator = generator
        self.max_instances: int = None
        self.vocab: Vocabulary = None
        self.padding_lengths: Dict[str, Dict[str, int]] = None
        self.num_instances = 0

        logger.info("initial pass through the dataset")
        fields_and_types: Dict[str, str] = None

        for instance in tqdm.tqdm(generator()):
            self.num_instances += 1

            # Check that all instances have the same "shape"
            instance_fields_and_types = {k: v.__class__.__name__ for k, v in instance.fields.items()}
            if fields_and_types is None:
                fields_and_types = instance_fields_and_types
            elif instance_fields_and_types != fields_and_types:
                raise ConfigurationError("You cannot construct a LazyDataset with non-homogeneous Instances.")


    def truncate(self, max_instances: int):
        """
        If there are more instances than ``max_instances`` in this dataset, we truncate the
        instances to the first ``max_instances``.  This `modifies` the current object, and returns
        nothing.
        """
        self.max_instances = max_instances

    def index_instances(self, vocab: Vocabulary):
        """
        This is a little bit misleading, as we have to index the instances as they're generated.
        However, code will still call this and it's a good place to grab a reference to the
        ``Vocabulary``.
        """
        self.vocab = vocab

        padding_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for instance in tqdm.tqdm(self.iterinstances()):
            instance.index_fields(vocab)
            instance_lengths = instance.get_padding_lengths()
            for field_name, instance_field_lengths in instance_lengths.items():
                for padding_key, length in instance_field_lengths.items():
                    prev_max_length = padding_lengths[field_name].get(padding_key, 0)
                    padding_lengths[field_name][padding_key] = max(length, prev_max_length)

        self.padding_lengths = {**padding_lengths}

    def iterinstances(self) -> Iterator[Instance]:
        for i, instance in tqdm.tqdm(enumerate(self.generator())):
            if self.max_instances is not None and i >= self.max_instances:
                return
            instance.index_fields(self.vocab)
            yield instance

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Gets the maximum padding lengths from all ``Instances`` in this dataset.  Each ``Instance``
        has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
        We look at all fields in all instances, and find the max values for each (field_name,
        padding_key) pair, returning them in a dictionary.

        This can then be used to convert this dataset into arrays of consistent length, or to set
        model parameters, etc.
        """
        return self.padding_lengths

    def as_tensor_dict(self,
                       padding_lengths: Dict[str, Dict[str, int]] = None,
                       cuda_device: int = -1,
                       for_training: bool = True,
                       verbose: bool = False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError("cannot call as_tensor_dict on a LazyDataset")
