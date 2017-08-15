from typing import Dict
from collections import defaultdict

from allennlp.data.fields.field import Field, DataArray


class ArrayField(Field[DataArray]):
    def __init__(self, array: DataArray) -> None:
        self.array = array

    def get_padding_lengths(self) -> Dict[str, int]:
        return defaultdict(int)

    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:  # pylint: disable=unused-argument
        return self.array
