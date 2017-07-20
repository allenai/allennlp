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

    def test_sort_tensor_by_length(self):
        tensor = torch.ones([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0

        sequence_lengths = torch.LongTensor([3, 4, 1, 5, 7])

        sorted_tensor, reverse_indices = tensor_utils.sort_batch_by_length(tensor, sequence_lengths)

        # Test sorted indices are padded correctly.
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].numpy(), 0.0)

        # Test restoration indices correctly recover the original tensor.
        assert sorted_tensor[reverse_indices].equal(tensor)
