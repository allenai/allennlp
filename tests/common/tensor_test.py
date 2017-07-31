# pylint: disable=invalid-name, no-self-use
import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import torch
from torch.autograd import Variable
import pytest

from allennlp.common.tensor import arrays_to_variables
from allennlp.common.tensor import get_lengths_from_binary_sequence_mask
from allennlp.common.tensor import masked_softmax
from allennlp.common.tensor import sort_batch_by_length
from allennlp.common.tensor import viterbi_decode
from allennlp.testing import AllenNlpTestCase


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
        sorted_tensor, sorted_lengths, reverse_indices = sort_batch_by_length(tensor, sequence_lengths)

        # Test sorted indices are padded correctly.
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].numpy(), 0.0)

        assert sorted_lengths.equal(torch.LongTensor([7, 5, 4, 3, 1]))

        # Test restoration indices correctly recover the original tensor.
        assert sorted_tensor[reverse_indices].equal(tensor)

    def test_masked_softmax_no_mask(self):
        # Testing the general unmasked 1D case.
        vector_1d = Variable(torch.FloatTensor([[1.0, 2.0, 3.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)

        vector_1d = Variable(torch.FloatTensor([[1.0, 2.0, 5.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.017148, 0.046613, 0.93624]]))

        # Testing the unmasked 1D case where the input is all 0s.
        vector_zero = Variable(torch.FloatTensor([[0.0, 0.0, 0.0]]))
        vector_zero_softmaxed = masked_softmax(vector_zero, None).data.numpy()
        assert_array_almost_equal(vector_zero_softmaxed,
                                  numpy.array([[0.33333334, 0.33333334, 0.33333334]]))

        # Testing the general unmasked batched case.
        matrix = Variable(torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.09003057, 0.24472847, 0.66524096]]))

        # Testing the unmasked batched case where one of the inputs are all 0s.
        matrix = Variable(torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self):
        # Testing the general masked 1D case.
        vector_1d = Variable(torch.FloatTensor([[1.0, 2.0, 5.0]]))
        mask_1d = Variable(torch.FloatTensor([[1.0, 0.0, 1.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382]]))

        vector_1d = Variable(torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = Variable(torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01321289, 0.0,
                                                0.26538793, 0.72139918]]))

        # Testing the masked 1D case where the input is all 0s and the mask
        # is not all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0, 0, 0, 1]]))

        # Testing the masked 1D case where the input is not all 0s
        # and the mask is all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0,
                                                0.0, 0.0]]))

        # Testing the masked 1D case where the input is all 0s and
        # the mask is all 0s.
        vector_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = Variable(torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_softmaxed = masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0,
                                                0.0, 0.0]]))

        # Testing the general masked batched case.
        matrix = Variable(torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s but
        # none of the masks are all 0.
        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s and
        # one of the masks are all 0.
        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.0, 0.0, 0.0]]))

        matrix = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
        masked_matrix_softmaxed = masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.0, 0.0, 0.0],
                                               [0.11920292, 0.0, 0.88079708]]))

    def test_viterbi_decode(self):
        # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
        sequence_predictions = torch.nn.functional.softmax(Variable(torch.rand([5, 9])))
        transition_matrix = torch.zeros([9, 9])
        indices, _ = viterbi_decode(sequence_predictions.data, transition_matrix)
        _, argmax_indices = torch.max(sequence_predictions, 1)
        assert indices == argmax_indices.data.squeeze().tolist()

        # Test that pairwise potentials effect the sequence correctly and that
        # viterbi_decode can handle -inf values.
        sequence_predictions = torch.FloatTensor([[0, 0, 0, 3, 4],
                                                  [0, 0, 0, 3, 4],
                                                  [0, 0, 0, 3, 4],
                                                  [0, 0, 0, 3, 4],
                                                  [0, 0, 0, 3, 4],
                                                  [0, 0, 0, 3, 4]])
        # The same tags shouldn't appear sequentially.
        transition_matrix = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float("-inf")
        indices, _ = viterbi_decode(sequence_predictions, transition_matrix)
        assert indices == [4, 3, 4, 3, 4, 3]

        # Test that unbalanced pairwise potentials break ties
        # between paths with equal unary potentials.
        sequence_predictions = torch.FloatTensor([[0, 0, 0, 4, 4],
                                                  [0, 0, 0, 4, 4],
                                                  [0, 0, 0, 4, 4],
                                                  [0, 0, 0, 4, 4],
                                                  [0, 0, 0, 4, 4],
                                                  [0, 0, 0, 4, 4]])
        # The 5th tag has a penalty for appearing sequentially.
        transition_matrix = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10

        indices, _ = viterbi_decode(sequence_predictions, transition_matrix)
        assert indices == [3, 3, 3, 3, 3, 3]
