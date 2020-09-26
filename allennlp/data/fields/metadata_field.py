from typing import Any, Dict, List, Mapping

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field


class MetadataField(Field[DataArray], Mapping[str, Any]):
    """
    A `MetadataField` is a `Field` that does not get converted into tensors.  It just carries
    side information that might be needed later on, for computing some third-party metric, or
    outputting debugging information, or whatever else you need.  We use this in the BiDAF model,
    for instance, to keep track of question IDs and passage token offsets, so we can more easily
    use the official evaluation script to compute metrics.

    We don't try to do any kind of smart combination of this field for batched input - when you use
    this `Field` in a model, you'll get a list of metadata objects, one for each instance in the
    batch.

    # Parameters

    metadata : `Any`
        Some object containing the metadata that you want to store.  It's likely that you'll want
        this to be a dictionary, but it could be anything you want.
    """

    __slots__ = ["metadata"]

    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata

    def __getitem__(self, key: str) -> Any:
        try:
            return self.metadata[key]  # type: ignore
        except TypeError:
            raise TypeError("your metadata is not a dict")

    def __iter__(self):
        try:
            return iter(self.metadata)
        except TypeError:
            raise TypeError("your metadata is not iterable")

    def __len__(self):
        try:
            return len(self.metadata)
        except TypeError:
            raise TypeError("your metadata has no length")

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:

        return self.metadata  # type: ignore

    @overrides
    def empty_field(self) -> "MetadataField":
        return MetadataField(None)

    @overrides
    def batch_tensors(self, tensor_list: List[DataArray]) -> List[DataArray]:  # type: ignore
        return tensor_list

    def __str__(self) -> str:
        return "MetadataField (print field.metadata to see specific information)."
