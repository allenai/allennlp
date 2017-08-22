from typing import Dict, List, Optional  # pylint: disable=unused-import
import logging

from overrides import overrides
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TagField(Field[numpy.ndarray]):
    """
    A ``TagField`` assigns a categorical label to each element in a :class:`SequenceField`.
    Because it's a labeling of some other field, we take that field as input here, and we use it to
    determine our padding and other things.

    This field will get converted into a sequence of one-hot vectors, where the size of each
    one-hot vector is the number of unique tags in your data.

    Parameters
    ----------
    tags : ``List[str]``
        A sequence of categorical labels, encoded as strings.  These could be POS tags like [NN,
        JJ, ...], BIO tags like [B-PERS, I-PERS, O, O, ...], or any other categorical tag sequence.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``TagField`` is labeling.  Most often, this is a
        ``TextField``, for tagging individual tokens in a sentence.
    tag_namespace : ``str``, optional (default='tags')
        The namespace to use for converting tag strings into integers.  We convert tag strings to
        integers for you, and this parameter tells the ``Vocabulary`` object which mapping from
        strings to integers to use (so that "O" as a tag doesn't get the same id as "O" as a word).
    """
    def __init__(self, tags: List[str], sequence_field: SequenceField, tag_namespace: str = 'tags') -> None:
        self._tags = tags
        self._sequence_field = sequence_field
        self._tag_namespace = tag_namespace
        self._indexed_tags = None  # type: Optional[List[int]]
        self._num_tags = None      # type: Optional[int]

        if not self._tag_namespace.endswith("tags"):
            logger.warning("Your tag namespace was '%s'. We recommend you use a namespace "
                           "ending with 'tags', so we don't add UNK and PAD tokens by "
                           "default to your vocabulary.  See documentation for "
                           "`non_padded_namespaces` parameter in Vocabulary.", self._tag_namespace)

        if len(tags) != sequence_field.sequence_length():
            raise ConfigurationError("Tag length and sequence length "
                                     "don't match: %d and %d" % (len(tags), sequence_field.sequence_length()))

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for tag in self._tags:
            counter[self._tag_namespace][tag] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_tags = [vocab.get_token_index(tag, self._tag_namespace) for tag in self._tags]
        self._num_tags = vocab.get_vocab_size(self._tag_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self._sequence_field.sequence_length()}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.ndarray:
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self._indexed_tags, desired_num_tokens)
        return padded_tags

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # pylint: disable=protected-access
        tag_field = TagField([], None)
        tag_field._indexed_tags = []
        return tag_field

    def tags(self):
        return self._tags
