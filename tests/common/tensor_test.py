# pylint: disable=invalid-name, no-self-use
import numpy
import torch
import pytest

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.tensor import arrays_to_variables
from allennlp.common.tensor import get_lengths_from_binary_sequence_mask
from allennlp.common.tensor import sort_batch_by_length


class TestTensor(AllenNlpTestCase):

    def test_data_structure_as_variables_handles_recursion(self):

        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4]),
                        "characters": numpy.ones([2, 5])
                        },
                "tags": numpy.ones([2, 3])
        }
        torch_array_dict = arrays_to_variables(array_dict)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.DoubleTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.DoubleTensor(numpy.ones([2, 5])))
        assert torch_array_dict["tags"].data.equal(
                torch.DoubleTensor(numpy.ones([2, 3])))

    def test_data_structure_as_variables_correctly_converts_mixed_types(self):

        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4], dtype="float32"),
                        "characters": numpy.ones([2, 5], dtype="int32")
                        },
                "tags": numpy.ones([2, 3], dtype="uint8")
        }
        torch_array_dict = arrays_to_variables(array_dict)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].data.equal(torch.ByteTensor(
                numpy.ones([2, 3], dtype="uint8")))

    @pytest.mark.skip
    def test_data_structure_as_variables_correctly_allocates_cuda_tensors(self):
        # TODO(Mark): Work out if we can test this somehow without actual GPUs.
        array_dict = {
                "sentence": {
                        "words": numpy.zeros([3, 4], dtype="float32"),
                        "characters": numpy.ones([2, 5], dtype="int32")
                        },
                "tags": numpy.ones([2, 3], dtype="uint8")
        }
        torch_array_dict = arrays_to_variables(array_dict, cuda_device=1)

        assert torch_array_dict["sentence"]["words"].data.equal(
                torch.cuda.FloatTensor(numpy.zeros([3, 4])))
        assert torch_array_dict["sentence"]["characters"].data.equal(
                torch.cuda.IntTensor(numpy.ones([2, 5], dtype="int32")))
        assert torch_array_dict["tags"].data.equal(
                torch.cuda.ByteTensor(numpy.ones([2, 3], dtype="uint8")))

    def test_get_sequence_lengths_from_binary_mask(self):
        binary_mask = torch.ByteTensor([[1, 1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0, 0]])
        lengths = get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.numpy(), numpy.array([3, 2, 6, 1]))

    def test_sort_tensor_by_length(self):
        tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0

        sequence_lengths = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor, reverse_indices = sort_batch_by_length(tensor, sequence_lengths)

        # Test sorted indices are padded correctly.
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].numpy(), 0.0)

        # Test restoration indices correctly recover the original tensor.
        assert sorted_tensor[reverse_indices].equal(tensor)
