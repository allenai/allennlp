from typing import Dict, List, Union
import logging

from overrides import overrides
import numpy

from .field import Field
from ..vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LabelField(Field):
    """
    A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
    text or 0-indexed integers.  If the labels need indexing, we will use a :class:`Vocabulary` to
    convert the string labels into integers.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default=``labels``)
        The namespace to use for converting label strings into integers.  If you have multiple
        different label fields in your data, you should make sure you use different namespaces for
        each one.
    num_labels : ``int``, optional (default=``None``)
        If your labels are 0-indexed integers, you can pass in the number of labels here, and we'll
        skip the indexing step.  If this is ``None``, no matter the type of ``label``, we'll use a
        vocabulary to give the labels new IDs.
    """
    def __init__(self,
                 label: Union[str, int],
                 label_namespace: str='*labels',
                 num_labels: int=None):
        self._label = label
        self._label_namespace = label_namespace
        if num_labels is None:
            self._label_id = None
            self._num_labels = None
            if not self._label_namespace.startswith("*"):
                logger.warning("The namespace of your tag (%s) does not begin with *,"
                               " meaning the vocabulary namespace will contain UNK "
                               "and PAD tokens by default.", self._label_namespace)
        else:
            assert isinstance(label, int), "Labels must be ints if you want to skip indexing"
            self._label_id = label
            self._num_labels = num_labels

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_id is None:
            counter[self._label_namespace][self._label] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_id is None:
            self._label_id = vocab.get_token_index(self._label, self._label_namespace)
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        label_array = numpy.zeros(self._num_labels)
        label_array[self._label_id] = 1
        return [label_array]

    @overrides
    def empty_field(self):
        return LabelField(0, self._label_namespace)

    def label(self):
        return self._label
