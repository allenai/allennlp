from typing import Dict


from allennlp.data.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary


class Instance:
    """
    An ``Instance`` is a collection of :class:`Field` objects, specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will end up as ``IndexedFields``, and will then be converted
    into padded arrays by a ``DataGenerator``.
    """
    def __init__(self, fields: Dict[str, Field]) -> None:
        self._fields = fields

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary items in all of the
        ``Fields`` in this ``Instance``.
        """
        for field in self._fields.values():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary):
        """
        Converts all ``UnindexedFields`` in this ``Instance`` to ``IndexedFields``, given the
        ``Vocabulary``.  This `mutates` the current object, it does not return a new ``Instance``.
        """
        for field in self._fields.values():
            field.index(vocab)

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name.  Each ``Field`` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for field_name, field in self._fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    def as_array(self, padding_lengths: Dict[str, Dict[str, int]]) -> Dict[str, DataArray]:
        """
        Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of numpy arrays for each field.
        """
        arrays = {}
        for field_name, field in self._fields.items():
            arrays[field_name] = field.as_array(padding_lengths[field_name])
        return arrays

    def fields(self):
        return self._fields
