from typing import Any, Dict, List, Mapping

from overrides import overrides

from allennlp.data.fields.field import DataArray, Field


class DetectronField(Field[DataArray], Mapping[str, Any]):
    """
    A `DetectronField` is a `Field` that does not get converted into tensors, because the Detectron2 models don't
    expect batched input.
    """

    __slots__ = ["image_dict"]

    def __init__(self, image_dict: Dict[str, Any]) -> None:
        self.image_dict = image_dict

    def __getitem__(self, key: str) -> Any:
        return self.image_dict[key]  # type: ignore

    def __iter__(self):
        return iter(self.image_dict)

    def __len__(self):
        return len(self.image_dict)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        return self.image_dict  # type: ignore

    @overrides
    def empty_field(self) -> "DetectronField":
        return DetectronField({})

    @overrides
    def batch_tensors(self, tensor_list: List[DataArray]) -> List[DataArray]:  # type: ignore
        return tensor_list

    def __str__(self) -> str:
        return "DetectronField(...)"
