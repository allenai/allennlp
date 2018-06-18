# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_allclose
import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TensorField


class TestLabelField(AllenNlpTestCase):
    def test_as_tensor(self):
        random_tensor = torch.rand(3, 5)
        tensorfield = TensorField(random_tensor)
        tensor = tensorfield.as_tensor(tensorfield.get_padding_lengths())
        assert isinstance(tensor, type(random_tensor))
        assert_allclose(tensor.cpu().numpy(), random_tensor.cpu().numpy())

    def test_as_tensor_returns_scalar(self):
        tensorfield = TensorField(torch.Tensor([5]))
        tensor = tensorfield.as_tensor(tensorfield.get_padding_lengths())
        assert tensor.item() == 5

    def test_tensor_field_raises_with_non_tensor_input(self):
        with pytest.raises(ConfigurationError):
            _ = TensorField("non tensor field")
        with pytest.raises(ConfigurationError):
            _ = TensorField([1, 2, 3])

    def test_tensor_field_empty_field_works(self):
        tensorfield = TensorField(torch.Tensor([1.0, 2.0]))
        empty_tensorfield = tensorfield.empty_field()
        assert isinstance(empty_tensorfield.tensor, type(torch.Tensor([])))
        assert_allclose(empty_tensorfield.tensor.cpu().numpy(),
                        torch.Tensor([]).cpu().numpy())

    def test_printing_doesnt_crash(self):
        tensorfield = TensorField(torch.Tensor([1.0, 2.0]))
        print(tensorfield)
