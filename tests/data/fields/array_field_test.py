import numpy
import torch

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data.fields import ArrayField, ListField


class TestArrayField(AllenNlpTestCase):
    def test_get_padding_lengths_correctly_returns_ordered_shape(self):
        shape = [3, 4, 5, 6]
        array = numpy.zeros(shape)
        array_field = ArrayField(array)
        lengths = array_field.get_padding_lengths()
        for i in range(len(lengths)):
            assert lengths["dimension_{}".format(i)] == shape[i]

    def test_as_tensor_handles_larger_padding_dimensions(self):
        shape = [3, 4]
        array = numpy.ones(shape)
        array_field = ArrayField(array)

        padded_tensor = (
            array_field.as_tensor({"dimension_0": 5, "dimension_1": 6}).detach().cpu().numpy()
        )
        numpy.testing.assert_array_equal(padded_tensor[:3, :4], array)
        numpy.testing.assert_array_equal(padded_tensor[3:, 4:], 0.0)

    def test_padding_handles_list_fields(self):
        array1 = ArrayField(numpy.ones([2, 3]))
        array2 = ArrayField(numpy.ones([1, 5]))
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = (
            list_field.as_tensor(list_field.get_padding_lengths()).detach().cpu().numpy()
        )
        correct_tensor = numpy.array(
            [
                [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            ]
        )
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)

    def test_padding_handles_list_fields_with_padding_values(self):
        array1 = ArrayField(numpy.ones([2, 3]), padding_value=-1)
        array2 = ArrayField(numpy.ones([1, 5]), padding_value=-1)
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = (
            list_field.as_tensor(list_field.get_padding_lengths()).detach().cpu().numpy()
        )
        correct_tensor = numpy.array(
            [
                [[1.0, 1.0, 1.0, -1.0, -1.0], [1.0, 1.0, 1.0, -1.0, -1.0]],
                [[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                [[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            ]
        )
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)

    def test_printing_doesnt_crash(self):
        array = ArrayField(numpy.ones([2, 3]), padding_value=-1)
        print(array)

    def test_as_tensor_works_with_scalar(self):
        array = ArrayField(numpy.asarray(42))
        returned_tensor = array.as_tensor(array.get_padding_lengths())
        current_tensor = numpy.asarray(42)
        numpy.testing.assert_array_equal(returned_tensor, current_tensor)

    def test_as_tensor_with_scalar_keeps_dtype(self):
        array = ArrayField(numpy.asarray(42, dtype=numpy.float32))
        returned_tensor = array.as_tensor(array.get_padding_lengths())
        assert returned_tensor.dtype == torch.float32

    def test_alternative_dtypes(self):
        shape = [3, 4, 5, 6]
        array = numpy.zeros(shape)

        # Setting dtype to numpy.int64 should produce a torch.LongTensor when field is converted to
        # a tensor
        array_field1 = ArrayField(array, dtype=numpy.int64)
        returned_tensor1 = array_field1.as_tensor(array_field1.get_padding_lengths())
        assert returned_tensor1.dtype == torch.int64

        # Setting dtype to numpy.uint8 should produce a torch.ByteTensor when field is converted to
        # a tensor
        array_field2 = ArrayField(array, dtype=numpy.uint8)
        returned_tensor2 = array_field2.as_tensor(array_field2.get_padding_lengths())
        assert returned_tensor2.dtype == torch.uint8

        # Padding should not affect dtype
        padding_lengths = {"dimension_" + str(i): 10 for i, _ in enumerate(shape)}
        padded_tensor = array_field2.as_tensor(padding_lengths)
        assert padded_tensor.dtype == torch.uint8

        # Empty fields should have the same dtype
        empty_field = array_field2.empty_field()
        assert empty_field.dtype == array_field2.dtype

    def test_len_works_with_scalar(self):
        array = ArrayField(numpy.asarray(42))
        assert len(array) == 1

    def test_eq(self):
        array1 = ArrayField(numpy.asarray([1, 1, 1]))
        array2 = ArrayField(numpy.asarray([[1, 1, 1], [1, 1, 1]]))
        array3 = ArrayField(numpy.asarray([1, 1, 2]))
        array4 = ArrayField(numpy.asarray([1, 1, 1]))
        assert array1 != array2
        assert array1 != array3
        assert array1 == array4
