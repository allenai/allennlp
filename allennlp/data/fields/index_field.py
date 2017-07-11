from typing import Dict, List

from overrides import overrides
import numpy

from .field import Field
from .sequence_field import SequenceField


class IndexField(Field):
    """
    An ``IndexField`` is an index into a :class:`SequenceField`, as might be used for
    representing a correct answer option in a list, or a span begin and span end position in a
    passage, for example.  Because it's an index into a :class:`SequenceField`, we take one of
    those as input and use it to compute padding lengths, so we create a one-hot representation of
    the correct length.

    An ``IndexField`` will get converted into a one-hot vector, where the size of the vector is the
    number of elements in the dependent ``SequenceField``.

    Parameters
    ----------
    index : ``int``
        The index to be represented as a 1 in the one-hot vector.  This is typically the "correct
        answer" in some classification decision over the sequence, like where an answer span starts
        in SQuAD, or which answer option is correct in a multiple choice question.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``IndexField`` is a pointer into.
    """
    def __init__(self, index: int, sequence_field: SequenceField):
        self._index = index
        self._sequence_field = sequence_field

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_options': self._sequence_field.sequence_length()}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        one_hot_index = numpy.zeros(padding_lengths['num_options'])
        one_hot_index[self._index] = 1
        return one_hot_index

    @overrides
    def empty_field(self):
        return IndexField(0, None)

    def sequence_field(self):
        return self._sequence_field

    def sequence_index(self):
        # This method can't be called index,
        # as that name is already reserved.
        return self._index
