# pylint: disable=no-self-use,invalid-name
import numpy

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

        padded_tensor = array_field.as_tensor({"dimension_0": 5, "dimension_1": 6}).data.cpu().numpy()
        numpy.testing.assert_array_equal(padded_tensor[:3, :4], array)
        numpy.testing.assert_array_equal(padded_tensor[3:, 4:], 0.)

    def test_padding_handles_list_fields(self):
        array1 = ArrayField(numpy.ones([2, 3]))
        array2 = ArrayField(numpy.ones([1, 5]))
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = list_field.as_tensor(list_field.get_padding_lengths()).data.cpu().numpy()
        correct_tensor = numpy.array([[[1., 1., 1., 0., 0.],
                                       [1., 1., 1., 0., 0.]],
                                      [[1., 1., 1., 1., 1.],
                                       [0., 0., 0., 0., 0.]],
                                      [[0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]])
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)

    def test_padding_handles_list_fields_with_padding_values(self):
        array1 = ArrayField(numpy.ones([2, 3]), padding_value=-1)
        array2 = ArrayField(numpy.ones([1, 5]), padding_value=-1)
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_tensor = list_field.as_tensor(list_field.get_padding_lengths()).data.cpu().numpy()
        correct_tensor = numpy.array([[[1., 1., 1., -1., -1.],
                                       [1., 1., 1., -1., -1.]],
                                      [[1., 1., 1., 1., 1.],
                                       [-1., -1., -1., -1., -1.]],
                                      [[-1., -1., -1., -1., -1.],
                                       [-1., -1., -1., -1., -1.]]])
        numpy.testing.assert_array_equal(returned_tensor, correct_tensor)
