from typing import Dict, Union, Sequence, Set, cast
import logging

from collections import defaultdict
from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MultiLabelField(Field[torch.Tensor]):
    """
    A ``MultiLabelField`` is an extension of the :class:`LabelField` that allows for multiple labels.
    It is particularly useful in multi-label classification where more than one label can be correct.
    As with the :class:`LabelField`, labels are either strings of text or 0-indexed integers (if you wish
    to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.

    This field will get converted into a vector of length equal to the vocabulary size with
    one hot encoding for the labels (all zeros, and ones for the labels).

    Parameters
    ----------
    labels : ``Sequence[Union[str, int]]``
    label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : ``bool``, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.
    """
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    # Dictionary from a vocab namespace to size. Need the vocab size to map labels to a tensor
    _vocab_size: Dict[str, int] = defaultdict(int)

    def __init__(self,
                 labels: Sequence[Union[str, int]],
                 label_namespace: str = 'labels',
                 skip_indexing: bool = False) -> None:
        self.labels = labels
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)

        if skip_indexing:
            if not all(isinstance(label, int) for label in labels):
                raise ConfigurationError("In order to skip indexing, your labels must be integers. "
                                         "Found labels = {}".format(labels))
            # vocabulary size = largest label id
            largest_label_id: int = cast(int, max(labels))
            MultiLabelField._vocab_size[self._label_namespace] = \
                    max(largest_label_id + 1, MultiLabelField._vocab_size[self._label_namespace])

            self._label_ids = labels
        else:
            if not all(isinstance(label, str) for label in labels):
                raise ConfigurationError("MultiLabelFields expects string labels if skip_indexing=False. "
                                         "Found labels: {}".format(labels))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith("labels") or label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_ids is None:
            self._label_ids = [vocab.get_token_index(label, self._label_namespace)  # type: ignore
                               for label in self.labels]

        # This is called after the full vocabulary has been built
        if MultiLabelField._vocab_size[self._label_namespace]:
            assert MultiLabelField._vocab_size[self._label_namespace] == vocab.get_vocab_size(
                    self._label_namespace), "Vocabulary size shouldn't change here."
        else:
            MultiLabelField._vocab_size[self._label_namespace] = vocab.get_vocab_size(self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> torch.Tensor:
        # pylint: disable=unused-argument

        values = torch.zeros(MultiLabelField._vocab_size[self._label_namespace])  # vector of zeros
        if self._label_ids:
            values.scatter_(0, torch.LongTensor(self._label_ids), 1)

        tensor = Variable(values, volatile=not for_training)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):
        return MultiLabelField([], self._label_namespace, skip_indexing=True)

    def __str__(self) -> str:
        return f"MultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"
