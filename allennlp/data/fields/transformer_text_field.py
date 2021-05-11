from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import torch.nn.functional

from allennlp.data.fields.field import Field
from allennlp.nn import util


class TransformerTextField(Field[torch.Tensor]):
    """
    A `TransformerTextField` is a collection of several tensors that are are a representation of text,
    tokenized and ready to become input to a transformer.
    """

    __slots__ = [
        "token_ids",
        "token_type_ids",
        "attention_mask",
        "special_tokens_mask",
        "offsets_mapping",
        "padding_token_id",
    ]

    def __init__(
        self,
        token_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        offsets_mapping: Optional[torch.Tensor] = None,
        padding_token_id: int = 0,
    ) -> None:
        self.token_ids = token_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.special_tokens_mask = special_tokens_mask
        self.offsets_mapping = offsets_mapping
        self.padding_token_id = padding_token_id

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {
            name: len(getattr(self, name))
            for name in self.__slots__
            if isinstance(getattr(self, name), torch.Tensor)
        }

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        result = {}
        for name, padding_length in padding_lengths.items():
            tensor = getattr(self, name)
            result[name] = torch.nn.functional.pad(
                tensor,
                (0, padding_length - len(tensor)),
                value=self.padding_token_id if name == "token_ids" else 0,
            )
        return result

    @overrides
    def empty_field(self):
        return TransformerTextField(torch.LongTensor())

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return util.batch_tensor_dicts(tensor_list)

    def human_readable_repr(self) -> Dict[str, Any]:
        def readable_tensor(t: torch.Tensor) -> str:
            if len(t) <= 16:
                return "[" + ", ".join(map(str, t)) + "]"
            else:
                return "[" + ", ".join(map(str, t[:8])) + ", ".join(map(str, t[-8:])) + "]"

        return {
            name: readable_tensor(getattr(self, name))
            for name in self.__slots__
            if isinstance(getattr(self, name), torch.Tensor)
        }
