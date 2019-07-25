from typing import Dict, List, Union, Set, Iterator
import logging
import textwrap

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SequenceLabelField(Field[torch.Tensor]):
    """
    A ``SequenceLabelField`` assigns a categorical label to each element in a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`.
    Because it's a labeling of some other field, we take that field as input here, and we use it to
    determine our padding and other things.

    This field will get converted into a list of integer class ids, representing the correct class
    for each element in the sequence.

    Parameters
    ----------
    labels : ``Union[List[str], List[int]]``
        A sequence of categorical labels, encoded as strings or integers.  These could be POS tags like [NN,
        JJ, ...], BIO tags like [B-PERS, I-PERS, O, O, ...], or any other categorical tag sequence. If the
        labels are encoded as integers, they will not be indexed using a vocab.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``SequenceLabelField`` is labeling.  Most often, this is a
        ``TextField``, for tagging individual tokens in a sentence.
    label_namespace : ``str``, optional (default='labels')
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the ``Vocabulary`` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    """
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 labels: Union[List[str], List[int]],
                 sequence_field: SequenceField,
                 label_namespace: str = 'labels') -> None:
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._indexed_labels = None
        self._maybe_warn_for_namespace(label_namespace)
        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError("Label length and sequence length "
                                     "don't match: %d and %d" % (len(labels), sequence_field.sequence_length()))

        if all([isinstance(x, int) for x in labels]):
            self._indexed_labels = labels

        elif not all([isinstance(x, str) for x in labels]):
            raise ConfigurationError("SequenceLabelFields must be passed either all "
                                     "strings or all ints. Found labels {} with "
                                     "types: {}.".format(labels, [type(x) for x in labels]))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    # Sequence methods
    def __iter__(self) -> Iterator[Union[str, int]]:
        return iter(self.labels)

    def __getitem__(self, idx: int) -> Union[str, int]:
        return self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_labels is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._indexed_labels is None:
            self._indexed_labels = [vocab.get_token_index(label, self._label_namespace)  # type: ignore
                                    for label in self.labels]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self._indexed_labels, desired_num_tokens)
        tensor = torch.LongTensor(padded_tags)
        return tensor

    @overrides
    def empty_field(self) -> 'SequenceLabelField':  # pylint: disable=no-self-use
        # pylint: disable=protected-access
        # The empty_list here is needed for mypy
        empty_list: List[str] = []
        sequence_label_field = SequenceLabelField(empty_list, self.sequence_field.empty_field())
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(["\t\t" + labels + "\n"
                                    for labels in textwrap.wrap(repr(self.labels), 100)])
        return f"SequenceLabelField of length {length} with " \
               f"labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."

    @overrides
    def __eq__(self, other):
        if isinstance(other, SequenceLabelField):
            return self.labels == other.labels
        return super().__eq__(other)
