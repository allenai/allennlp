# pylint: disable=no-self-use
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length


class ListField(SequenceField[DataArray]):
    """
    A ``ListField`` is a list of other fields.  You would use this to represent, e.g., a list of
    answer options that are themselves ``TextFields``.

    This field will get converted into a tensor that has one more mode than the items in the list.
    If this is a list of ``TextFields`` that have shape (num_words, num_characters), this
    ``ListField`` will output a tensor of shape (num_sentences, num_words, num_characters).

    Parameters
    ----------
    field_list : ``List[Field]``
        A list of ``Field`` objects to be concatenated into a single input tensor.  All of the
        contained ``Field`` objects must be of the same type.
    """
    def __init__(self, field_list: List[Field]) -> None:
        field_class_set = set([field.__class__ for field in field_list])
        assert len(field_class_set) == 1, "ListFields must contain a single field type, found " +\
                                          str(field_class_set)
        self.field_list = field_list  # type: List[Field]

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for field in self.field_list:
            field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        for field in self.field_list:
            field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        field_lengths = [field.get_padding_lengths() for field in self.field_list]
        padding_lengths = {'num_fields': len(self.field_list)}
        for key in field_lengths[0].keys():
            padding_lengths[key] = max(x[key] if key in x else 0 for x in field_lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self.field_list)

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        padded_field_list = pad_sequence_to_length(self.field_list,
                                                   padding_lengths['num_fields'],
                                                   self.field_list[0].empty_field)
        padded_fields = [field.as_array(padding_lengths) for field in padded_field_list]
        if isinstance(padded_fields[0], dict):
            namespaces = list(padded_fields[0].keys())
            return {namespace: numpy.array([field[namespace] for field in padded_fields])
                    for namespace in namespaces}
        else:
            return numpy.asarray(padded_fields)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        raise RuntimeError("Nested ListFields are not implemented, and if you want this "
                           "you should probably try to simplify your data type, anyway")
