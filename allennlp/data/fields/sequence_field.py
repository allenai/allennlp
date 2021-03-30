from allennlp.data.fields.field import DataArray, Field


class SequenceField(Field[DataArray]):
    """
    A `SequenceField` represents a sequence of things.  This class just adds a method onto
    `Field`: :func:`sequence_length`.  It exists so that `SequenceLabelField`, `IndexField` and other
    similar `Fields` can have a single type to require, with a consistent API, whether they are
    pointing to words in a `TextField`, items in a `ListField`, or something else.
    """

    __slots__ = []  # type: ignore

    def sequence_length(self) -> int:
        """
        How many elements are there in this sequence?
        """
        raise NotImplementedError

    def empty_field(self) -> "SequenceField":
        raise NotImplementedError
