from typing import Dict
import logging

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BooleanField(Field[numpy.ndarray]):
    """
    A ``BooleanField`` is a boolean label of some kind, where the labels are either True or False.

    Parameters
    ----------
    bool_label : ``Union[str, int]``.
    """
    def __init__(self,
                 label: bool):
        self.label = label

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.ndarray:  # pylint: disable=unused-argument
        return numpy.asarray([self.label])

    @overrides
    def empty_field(self):
        return BooleanField(0)
