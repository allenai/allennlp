import pytest
import torch

from allennlp.modules import Attention
from allennlp.modules.attention import BilinearAttention, AdditiveAttention, LinearAttention


@pytest.mark.parametrize("attention_type", Attention.list_available())
def test_all_attention_works_the_same(attention_type: str):
    module_cls = Attention.by_name(attention_type)

    vector = torch.FloatTensor([[-7, -8, -9]])
    matrix = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]]])

    if module_cls in {BilinearAttention, AdditiveAttention, LinearAttention}:
        module = module_cls(vector.size(-1), matrix.size(-1))
    else:
        module = module_cls()

    output = module(vector, matrix)
    assert tuple(output.size()) == (1, 2)
