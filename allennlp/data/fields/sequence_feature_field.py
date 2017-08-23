# pylint: disable=no-self-use
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length
from allennlp.common.checks import ConfigurationError


class SequenceFeatureField(Field[numpy.ndarray]):
    """
    A ``SequenceFeatureField`` is an optional index into a :class:`SequenceField`, as might be used for
    representing a correct answer option in a list, or a span begin and span end position in a
    passage, for example.  Because the features depend on the presence of a :class:`SequenceField`,
    we take one as input and use it to specify the padding lengths, so the features are padded/truncated
    to the same length.

    A ``SequenceFeatureField`` will get converted into a 1D :class:`numpy.array`, where the
    size of the vector is the number of elements in the dependent ``SequenceField``.

    Parameters
    ----------
    feature_indices : ``List[int]``
        The per token indices to be represented as features.
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``SequenceFeatureField`` has features for.
    """
    def __init__(self, feature_indices: List[int], sequence_field: SequenceField) -> None:
        self._feature_indices = feature_indices
        self._sequence_field = sequence_field

        if len(feature_indices) != sequence_field.sequence_length():
            raise ConfigurationError("Feature indices and sequence length "
                                     "don't match: %d and %d" % (len(feature_indices),
                                                                 sequence_field.sequence_length()))

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_options': self._sequence_field.sequence_length()}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.array:
        sequence_length = padding_lengths["num_options"]
        padded_features = pad_sequence_to_length(self._feature_indices, sequence_length)
        return numpy.asarray(padded_features)

    @overrides
    def empty_field(self):
        return SequenceFeatureField(None, None)

    def sequence_field(self):
        return self._sequence_field

    def feature_indices(self):
        return self._feature_indices
