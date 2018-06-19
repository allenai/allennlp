# pylint: disable=no-self-use,invalid-name
import numpy
import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields import ArrayField, ListField

class TestArrayField():

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_get_padding_lengths_correctly_returns_ordered_shape(self, array_type):
        shape = [3, 4, 5, 6]
        array = numpy.zeros(shape) if array_type == "numpy" else torch.zeros(shape)
        array_field = ArrayField(array)
        lengths = array_field.get_padding_lengths()
        for i in range(len(lengths)):
            assert lengths["dimension_{}".format(i)] == shape[i]

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_as_tensor_handles_larger_padding_dimensions(self, array_type):
        shape = [3, 4]
        array = numpy.ones(shape) if array_type == "numpy" else torch.ones(shape)
        array_field = ArrayField(array)

        padded_tensor = array_field.as_tensor({"dimension_0": 5, "dimension_1": 6}).detach().cpu().numpy()
        numpy.testing.assert_array_equal(padded_tensor[:3, :4], array)
        numpy.testing.assert_array_equal(padded_tensor[3:, 4:], 0.)

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_padding_handles_list_fields(self, array_type):
        ones_function = numpy.ones if array_type == "numpy" else torch.ones
        array1 = ArrayField(ones_function([2, 3]))
        array2 = ArrayField(ones_function([1, 5]))
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = list_field.as_tensor(list_field.get_padding_lengths()).detach().cpu().numpy()
        correct_tensor = numpy.array([[[1., 1., 1., 0., 0.],
                                       [1., 1., 1., 0., 0.]],
                                      [[1., 1., 1., 1., 1.],
                                       [0., 0., 0., 0., 0.]],
                                      [[0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]])
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_padding_handles_list_fields_with_padding_values(self, array_type):
        ones_function = numpy.ones if array_type == "numpy" else torch.ones
        array1 = ArrayField(ones_function([2, 3]), padding_value=-1)
        array2 = ArrayField(ones_function([1, 5]), padding_value=-1)
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = list_field.as_tensor(list_field.get_padding_lengths()).detach().cpu().numpy()
        correct_tensor = numpy.array([[[1., 1., 1., -1., -1.],
                                       [1., 1., 1., -1., -1.]],
                                      [[1., 1., 1., 1., 1.],
                                       [-1., -1., -1., -1., -1.]],
                                      [[-1., -1., -1., -1., -1.],
                                       [-1., -1., -1., -1., -1.]]])
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)

    def test_raises_with_non_tensor_input(self):
        with pytest.raises(ConfigurationError):
            _ = ArrayField("non array field")
        with pytest.raises(ConfigurationError):
            _ = ArrayField([1, 2, 3])

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_batch_tensors(self, array_type):
        ones_function = numpy.ones if array_type == "numpy" else torch.ones
        zeros_function = numpy.zeros if array_type == "numpy" else torch.zeros
        field = ArrayField(ones_function(3))
        tensor1 = field.as_tensor(field.get_padding_lengths())

        field = ArrayField(zeros_function(3))
        tensor2 = field.as_tensor(field.get_padding_lengths())
        tensor_list = [tensor1, tensor2]
        numpy.testing.assert_allclose(field.batch_tensors(tensor_list).cpu().numpy(),
                                      torch.stack(tensor_list).cpu().numpy())

    @pytest.mark.parametrize("array_type", ("numpy", "torch"))
    def test_printing_doesnt_crash(self, array_type):
        data = numpy.ones([2, 3]) if array_type == "numpy" else torch.ones([2, 3])
        array = ArrayField(data, padding_value=-1)
        print(array)
