from typing import Dict, Union
import logging

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LabelField(Field[numpy.array]):
    """
    A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
    text or 0-indexed integers.  If the labels need indexing, we will use a :class:`Vocabulary` to
    convert the string labels into integers.

    This field will get converted into a one-hot vector, where the size of the vector is the number
    of unique labels in your data.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default=``labels``)
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to one-hot vectors),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    num_labels : ``int``, optional (default=``None``)
        If your labels are 0-indexed integers, you can pass in the number of labels here, and we'll
        skip the indexing step.  If this is ``None``, no matter the type of ``label``, we'll use a
        vocabulary to give the labels new IDs.
    """
    def __init__(self,
                 label: Union[str, int],
                 label_namespace: str = 'labels',
                 num_labels: int = None) -> None:
        self._label = label
        self._label_namespace = label_namespace
        if num_labels is None:
            self._label_id = None
            self._num_labels = None
            if not self._label_namespace.endswith("labels"):
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.", self._label_namespace)
        else:
            assert isinstance(label, int), "Labels must be ints if you want to skip indexing"
            self._label_id = label
            self._num_labels = num_labels

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_id is None:
            counter[self._label_namespace][self._label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_id is None:
            self._label_id = vocab.get_token_index(self._label, self._label_namespace)  # type: ignore
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.array:  # pylint: disable=unused-argument
        label_array = numpy.zeros(self._num_labels)
        label_array[self._label_id] = 1
        return label_array

    @overrides
    def empty_field(self):
        return LabelField(0, self._label_namespace, self._num_labels)

    def label(self):
        return self._label
