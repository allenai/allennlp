# pylint: disable=no-self-use
from typing import Any, Dict, List

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field


class MetadataField(Field[DataArray]):
    """
    A ``MetadataField`` is a ``Field`` that does not get converted into arrays.  It just carries
    side information that might be needed later on, for computing some third-party metric, or
    outputting debugging information, or whatever else you need.  We use this in the BiDAF model,
    for instance, to keep track of question IDs and passage token offsets, so we can more easily
    use the official evaluation script to compute metrics.

    We don't try to do any kind of smart combination of this field for batched input - when you use
    this ``Field`` in a model, you'll get a list of metadata objects, one for each instance in the
    batch.

    Note that if you use this field, you are `required` to include ``metadata`` in the field name
    used as a key in ``Instance``.  Otherwise we won't know to treat the output of this field
    specially in :func:`~allennlp.nn.util.arrays_to_variables`.

    Parameters
    ----------
    metadata : ``Any``
        Some object containing the metadata that you want to store.  It's likely that you'll want
        this to be a dictionary, but it could be anything you want.
    """
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        # pylint: disable=unused-argument
        return self.metadata  # type: ignore

    @overrides
    def empty_field(self) -> 'MetadataField':
        return MetadataField(None)

    @classmethod
    @overrides
    def batch_arrays(cls, array_list: List[DataArray]) -> DataArray:  # type: ignore
        return array_list  # type: ignore
