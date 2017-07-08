import torch
import numpy
from allennlp.testing.test_case import AllenNlpTestCase
import allennlp.common.tensor as tensor_utils


class TestTensorUtils(AllenNlpTestCase):

    def test_get_sequence_lengths_from_binary_mask(self):
        binary_mask = torch.ByteTensor([[1, 1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0, 0]])
        lengths = tensor_utils.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.numpy(), numpy.array([3, 2, 6, 1]))

    def test_get_sequence_lengths_from_sequence_tensor(self):
        tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0
        mask = tensor_utils.get_lengths_from_sequence_tensor(tensor)
        numpy.testing.assert_array_equal(mask.numpy(), numpy.array([3, 4, 1, 5, 7]))
