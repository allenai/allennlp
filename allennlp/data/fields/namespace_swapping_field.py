from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data.fields.field import Field


class NamespaceSwappingField(Field[torch.Tensor]):
    """
    A `NamespaceSwappingField` is used to map tokens in one namespace to tokens in another namespace.
    It is used by seq2seq models with a copy mechanism that copies tokens from the source
    sentence into the target sentence.

    # Parameters

    source_tokens : `List[Token]`
        The tokens from the source sentence.
    target_namespace : `str`
        The namespace that the tokens from the source sentence will be mapped to.
    """

    __slots__ = ["_source_tokens", "_target_namespace", "_mapping_array"]

    def __init__(self, source_tokens: List[Token], target_namespace: str) -> None:
        self._source_tokens = source_tokens
        self._target_namespace = target_namespace
        self._mapping_array: List[int] = []

    @overrides
    def index(self, vocab: Vocabulary):
        self._mapping_array = [
            vocab.get_token_index(x.ensure_text(), self._target_namespace)
            for x in self._source_tokens
        ]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": len(self._source_tokens)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_length = padding_lengths["num_tokens"]
        padded_tokens = pad_sequence_to_length(self._mapping_array, desired_length)
        tensor = torch.LongTensor(padded_tokens)
        return tensor

    @overrides
    def empty_field(self) -> "NamespaceSwappingField":
        empty_field = NamespaceSwappingField([], self._target_namespace)
        empty_field._mapping_array = []

        return empty_field

    def __len__(self):
        return len(self._source_tokens)
