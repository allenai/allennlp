"""
A :class:`~Dataset` represents a collection of data suitable for feeding into a model.
For example, when you train a model, you will likely have a *training* dataset and a *validation* dataset.
"""

import logging
from typing import Iterator, Iterable

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Dataset(Iterable):
    def __init__(self, num_instances: int = None):
        self.num_instances = num_instances

    def __iter__(self) -> Iterator[Instance]:
        raise NotImplementedError

    def index_instances(self, vocab: Vocabulary):
        """
        Converts all ``UnindexedFields`` in all ``Instances`` in this ``Dataset`` into
        ``IndexedFields``.  This modifies the current object, it does not return a new object.
        Depending on the subclass implementation, this indexing may occur immediately, or it
        may not occur until iteration time.
        """
        raise NotImplementedError
