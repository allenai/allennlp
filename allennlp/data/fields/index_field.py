# pylint: disable=no-self-use
from typing import Dict, Optional

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField


class IndexField(Field[numpy.array]):
    """
    An ``IndexField`` is an optional index into a :class:`SequenceField`, as might be used for
    representing a correct answer option in a list, or a span begin and span end position in a
    passage, for example.  Because it's an index into a :class:`SequenceField`, we take one of
    those as input and use it to compute padding lengths, so we create an optional one-hot
    representation of the correct length. As the index is optional, this Field can be used as
    a binary indicator for inputs which are optional.

    An ``IndexField`` will get converted into a one-hot vector, where the size of the vector is the
    number of elements in the dependent ``SequenceField``.

    Parameters
    ----------
    index : ``Optional[int]``
        The index to be represented as a 1 in the one-hot vector.  This is typically the "correct
        answer" in some classification decision over the sequence, like where an answer span starts
        in SQuAD, or which answer option is correct in a multiple choice question. If the index is
        is ``None``, an array of all zeros is returned.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``IndexField`` is a pointer into.
    """
    def __init__(self, index: Optional[int], sequence_field: SequenceField) -> None:
        self._index = index
        self._sequence_field = sequence_field

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_options': self._sequence_field.sequence_length()}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.array:
        index = numpy.zeros(padding_lengths['num_options'])
        if self._index is None:
            return index
        index[self._index] = 1
        return index

    @overrides
    def empty_field(self):
        return IndexField(None, None)

    def sequence_field(self):
        return self._sequence_field

    def sequence_index(self):
        # This method can't be called index,
        # as that name is already reserved.
        return self._index
