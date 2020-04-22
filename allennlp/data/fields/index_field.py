from typing import Dict

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.checks import ConfigurationError


class IndexField(Field[torch.Tensor]):
    """
    An `IndexField` is an index into a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`, as might be used for representing
    a correct answer option in a list, or a span begin and span end position in a passage, for
    example.  Because it's an index into a :class:`SequenceField`, we take one of those as input
    and use it to compute padding lengths.

    # Parameters

    index : `int`
        The index of the answer in the :class:`SequenceField`.  This is typically the "correct
        answer" in some classification decision over the sequence, like where an answer span starts
        in SQuAD, or which answer option is correct in a multiple choice question.  A value of
        `-1` means there is no label, which can be used for padding or other purposes.
    sequence_field : `SequenceField`
        A field containing the sequence that this `IndexField` is a pointer into.
    """

    def __init__(self, index: int, sequence_field: SequenceField) -> None:
        self.sequence_index = index
        self.sequence_field = sequence_field

        if not isinstance(index, int):
            raise ConfigurationError(
                "IndexFields must be passed integer indices. "
                "Found index: {} with type: {}.".format(index, type(index))
            )

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:

        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:

        tensor = torch.LongTensor([self.sequence_index])
        return tensor

    @overrides
    def empty_field(self):
        return IndexField(-1, self.sequence_field.empty_field())

    def __str__(self) -> str:
        return f"IndexField with index: {self.sequence_index}."

    def __eq__(self, other) -> bool:
        # Allow equality checks to ints that are the sequence index
        if isinstance(other, int):
            return self.sequence_index == other
        return super().__eq__(other)

    def __len__(self):
        return 1
