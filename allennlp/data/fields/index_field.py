# pylint: disable=no-self-use
from typing import Dict

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.checks import ConfigurationError


class IndexField(Field[numpy.ndarray]):
    """
    An ``IndexField`` is an optional index into a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`, as might be used for
    representing a correct answer option in a list, or a span begin and span end position in a
    passage, for example.  Because it's an index into a :class:`SequenceField`, we take one of
    those as input and use it to compute padding lengths.

    Parameters
    ----------
    index : ``Optional[int]``
        The index of the answer in the :class:`SequenceField`.  This is typically the "correct
        answer" in some classification decision over the sequence, like where an answer span starts
        in SQuAD, or which answer option is correct in a multiple choice question.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``IndexField`` is a pointer into.
    """
    def __init__(self, index: int, sequence_field: SequenceField) -> None:
        self.sequence_index = index
        self.sequence_field = sequence_field

        if not isinstance(index, int):
            raise ConfigurationError("IndexFields must be passed integer indices. "
                                     "Found index: {} with type: {}.".format(index, type(index)))

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_options': self.sequence_field.sequence_length()}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.array:  # pylint: disable=unused-argument
        return numpy.asarray([self.sequence_index])

    @overrides
    def empty_field(self):
        return IndexField(None, None)
