from typing import Generic
from allennlp.data.fields.field import Field, DataArray


class SequenceField(Field[DataArray], Generic[DataArray]):
    """
    A ``SequenceField`` represents a sequence of things.  This class just adds a method onto
    ``Field``: :func:`sequence_length`.  It exists so that ``TagField``, ``IndexField`` and other
    similar ``Fields`` can have a single type to require, with a consistent API, whether they are
    pointing to words in a ``TextField``, items in a ``ListField``, or something else.
    """
    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError
