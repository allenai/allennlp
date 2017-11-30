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

    def test_as_array_handles_larger_padding_dimensions(self):
        shape = [3, 4]
        array = numpy.ones(shape)
        array_field = ArrayField(array)

        padded_array = array_field.as_array({"dimension_0": 5, "dimension_1": 6})
        numpy.testing.assert_array_equal(padded_array[:3, :4], array)
        numpy.testing.assert_array_equal(padded_array[3:, 4:], 0.)

    def test_padding_handles_list_fields(self):
        array1 = ArrayField(numpy.ones([2, 3]))
        array2 = ArrayField(numpy.ones([1, 5]))
        empty_array = array1.empty_field()
        list_field = ListField([array1, array2, empty_array])

        returned_array = list_field.as_array(list_field.get_padding_lengths())
        correct_array = numpy.array([[[1., 1., 1., 0., 0.],
                                      [1., 1., 1., 0., 0.]],
                                     [[1., 1., 1., 1., 1.],
                                      [0., 0., 0., 0., 0.]],
                                     [[0., 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0.]]])
        numpy.testing.assert_array_equal(returned_array, correct_array)
