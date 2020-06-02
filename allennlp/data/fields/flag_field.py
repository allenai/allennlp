from typing import Any, Dict, List

from overrides import overrides

from allennlp.data.fields.field import Field


class FlagField(Field[Any]):
    """
    A class representing a flag, which must be constant across all instances in a batch.
    This will be passed to a `forward` method as a single value of whatever type you pass in.
    """

    __slots__ = ["flag_value"]

    def __init__(self, flag_value: Any) -> None:
        self.flag_value = flag_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Any:
        return self.flag_value

    @overrides
    def empty_field(self):
        # Because this has to be constant across all instances in a batch, we need to keep the same
        # value.
        return FlagField(self.flag_value)

    def __str__(self) -> str:
        return f"FlagField({self.flag_value})"

    def __len__(self) -> int:
        return 1

    @overrides
    def batch_tensors(self, tensor_list: List[Any]) -> Any:
        if len(set(tensor_list)) != 1:
            raise ValueError(
                f"Got different values in a FlagField when trying to batch them: {tensor_list}"
            )
        return tensor_list[0]
