import pytest
import torch

from allennlp.modules import MatrixAttention
from allennlp.modules.matrix_attention import BilinearMatrixAttention, LinearMatrixAttention


@pytest.mark.parametrize("attention_type", MatrixAttention.list_available())
def test_all_attention_works_the_same(attention_type: str):
    module_cls = MatrixAttention.by_name(attention_type)

    matrix1 = torch.FloatTensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    matrix2 = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]]])

    if module_cls in {BilinearMatrixAttention, LinearMatrixAttention}:
        module = module_cls(matrix1.size(-1), matrix2.size(-1))
    else:
        module = module_cls()

    output = module(matrix1, matrix2)
    assert tuple(output.size()) == (1, 3, 2)
