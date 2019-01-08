# pylint: disable=invalid-name,no-self-use,too-many-public-methods,not-callable
import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import torch
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn import util


class TestNnUtil(AllenNlpTestCase):
    def test_get_sequence_lengths_from_binary_mask(self):
        binary_mask = torch.ByteTensor([[1, 1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0, 0]])
        lengths = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.numpy(), numpy.array([3, 2, 6, 1]))

    def test_get_mask_from_sequence_lengths(self):
        sequence_lengths = torch.LongTensor([4, 3, 1, 4, 2])
        mask = util.get_mask_from_sequence_lengths(sequence_lengths, 5).data.numpy()
        assert_almost_equal(mask, [[1, 1, 1, 1, 0],
                                   [1, 1, 1, 0, 0],
                                   [1, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 0],
                                   [1, 1, 0, 0, 0]])

    def test_get_sequence_lengths_converts_to_long_tensor_and_avoids_variable_overflow(self):
        # Tests the following weird behaviour in Pytorch 0.1.12
        # doesn't happen for our sequence masks:
        #
        # mask = torch.ones([260]).byte()
        # mask.sum() # equals 260.
        # var_mask = t.a.V(mask)
        # var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
        binary_mask = torch.ones(2, 260).byte()
        lengths = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.data.numpy(), numpy.array([260, 260]))

    def test_sort_tensor_by_length(self):
        tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0

        sequence_lengths = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor, sorted_lengths, reverse_indices, _ = util.sort_batch_by_length(tensor, sequence_lengths)

        # Test sorted indices are padded correctly.
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].data.numpy(), 0.0)

        assert sorted_lengths.data.equal(torch.LongTensor([7, 5, 4, 3, 1]))

        # Test restoration indices correctly recover the original tensor.
        assert sorted_tensor.index_select(0, reverse_indices).data.equal(tensor.data)

    def test_get_final_encoder_states(self):
        encoder_outputs = torch.Tensor([[[1, 2, 3, 4],
                                         [5, 6, 7, 8],
                                         [9, 10, 11, 12]],
                                        [[13, 14, 15, 16],
                                         [17, 18, 19, 20],
                                         [21, 22, 23, 24]]])
        mask = torch.Tensor([[1, 1, 1], [1, 1, 0]])
        final_states = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=False)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 11, 12], [17, 18, 19, 20]])
        final_states = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=True)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 3, 4], [17, 18, 15, 16]])

    def test_masked_softmax_no_mask(self):
        # Testing the general unmasked 1D case.
        vector_1d = torch.FloatTensor([[1.0, 2.0, 3.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)

        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.017148, 0.046613, 0.93624]]))

        # Testing the unmasked 1D case where the input is all 0s.
        vector_zero = torch.FloatTensor([[0.0, 0.0, 0.0]])
        vector_zero_softmaxed = util.masked_softmax(vector_zero, None).data.numpy()
        assert_array_almost_equal(vector_zero_softmaxed,
                                  numpy.array([[0.33333334, 0.33333334, 0.33333334]]))

        # Testing the general unmasked batched case.
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.09003057, 0.24472847, 0.66524096]]))

        # Testing the unmasked batched case where one of the inputs are all 0s.
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self):
        # Testing the general masked 1D case.
        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d = torch.FloatTensor([[1.0, 0.0, 1.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382]]))

        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))

        # Testing the masked 1D case where the input is all 0s and the mask
        # is not all 0s.
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0, 0, 0, 1]]))

        # Testing the masked 1D case where the input is not all 0s
        # and the mask is all 0s.
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0, 0.0, 0.0]]))

        # Testing the masked 1D case where the input is all 0s and
        # the mask is all 0s.
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0, 0.0, 0.0]]))

        # Testing the masked 1D case where there are large elements in the
        # padding.
        vector_1d = torch.FloatTensor([[1.0, 1.0, 1e5]])
        mask_1d = torch.FloatTensor([[1.0, 1.0, 0.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.5, 0.5, 0]]))

        # Testing the general masked batched case.
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s but
        # none of the masks are all 0.
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s and
        # one of the masks are all 0.
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.0, 0.0, 0.0]]))

        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.0, 0.0, 0.0],
                                               [0.11920292, 0.0, 0.88079708]]))

    def test_masked_log_softmax_masked(self):
        # Tests replicated from test_softmax_masked - we test that exponentiated,
        # the log softmax contains the correct elements (masked elements should be == 1).

        # Testing the general masked 1D case.
        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d = torch.FloatTensor([[1.0, 0.0, 1.0]])
        vector_1d_softmaxed = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed),
                                  numpy.array([[0.01798621, 0.0, 0.98201382]]))

        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]])
        vector_1d_softmaxed = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed),
                                  numpy.array([[0.01321289, 0.0,
                                                0.26538793, 0.72139918]]))

        # Testing the masked 1D case where the input is all 0s and the mask
        # is not all 0s.
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]])
        vector_1d_softmaxed = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed),
                                  numpy.array([[0., 0., 0., 1.]]))

        # Testing the masked 1D case where the input is not all 0s
        # and the mask is all 0s.  The output here will be arbitrary, but it should not be nan.
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        vector_1d_softmaxed = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert not numpy.isnan(vector_1d_softmaxed).any()

    def test_masked_max(self):
        # Testing the general masked 1D case.
        vector_1d = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d = torch.FloatTensor([1.0, 0.0, 1.0])
        vector_1d_maxed = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_maxed, 5.0)

        # Testing if all masks are zero, the output will be arbitrary, but it should not be nan.
        vector_1d = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d = torch.FloatTensor([0.0, 0.0, 0.0])
        vector_1d_maxed = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_maxed).any()

        # Testing batch value and batch masks
        matrix = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        matrix_maxed = util.masked_max(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([5.0, -1.0]))

        # Testing keepdim for batch value and batch masks
        matrix = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        matrix_maxed = util.masked_max(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0], [-1.0]]))

        # Testing broadcast
        matrix = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]],
                                    [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).unsqueeze(-1)
        matrix_maxed = util.masked_max(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0, 2.0], [-1.0, -0.5]]))

    def test_masked_mean(self):
        # Testing the general masked 1D case.
        vector_1d = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d = torch.FloatTensor([1.0, 0.0, 1.0])
        vector_1d_mean = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_mean, 3.0)

        # Testing if all masks are zero, the output will be arbitrary, but it should not be nan.
        vector_1d = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d = torch.FloatTensor([0.0, 0.0, 0.0])
        vector_1d_mean = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_mean).any()

        # Testing batch value and batch masks
        matrix = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        matrix_mean = util.masked_mean(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([3.0, -1.5]))

        # Testing keepdim for batch value and batch masks
        matrix = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        matrix_mean = util.masked_mean(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0], [-1.5]]))

        # Testing broadcast
        matrix = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]],
                                    [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask = torch.FloatTensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]).unsqueeze(-1)
        matrix_mean = util.masked_mean(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0, 0.5], [-1.5, -1.75]]))

    def test_masked_flip(self):
        tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [5, 5, 5]]])
        solution = [[[6, 6, 6], [0, 0, 0]], [[4, 4, 4], [3, 3, 3]]]
        response = util.masked_flip(tensor, [1, 2])
        assert_almost_equal(response, solution)

        tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]],
                                    [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]]])
        solution = [[[2, 2, 2], [1, 1, 1], [6, 6, 6], [0, 0, 0]],
                    [[1, 2, 3], [5, 5, 5], [4, 4, 4], [3, 3, 3]]]
        response = util.masked_flip(tensor, [3, 4])
        assert_almost_equal(response, solution)

        tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]],
                                    [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]],
                                    [[1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0]]])
        solution = [[[2, 2, 2], [1, 1, 1], [6, 6, 6], [0, 0, 0]],
                    [[1, 2, 3], [5, 5, 5], [4, 4, 4], [3, 3, 3]],
                    [[2, 2, 2], [1, 1, 1], [0, 0, 0], [0, 0, 0]]]
        response = util.masked_flip(tensor, [3, 4, 2])
        assert_almost_equal(response, solution)

    def test_get_text_field_mask_returns_a_correct_mask(self):
        text_field_tensors = {
                "tokens": torch.LongTensor([[3, 4, 5, 0, 0], [1, 2, 0, 0, 0]]),
                "token_characters": torch.LongTensor([[[1, 2], [3, 0], [2, 0], [0, 0], [0, 0]],
                                                      [[5, 0], [4, 6], [0, 0], [0, 0], [0, 0]]])
        }

        assert_almost_equal(util.get_text_field_mask(text_field_tensors).numpy(),
                            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_character_only_input(self):
        text_field_tensors = {
                "token_characters": torch.LongTensor([[[1, 2, 3], [3, 0, 1], [2, 1, 0], [0, 0, 0]],
                                                      [[5, 5, 5], [4, 6, 0], [0, 0, 0], [0, 0, 0]]])
        }
        assert_almost_equal(util.get_text_field_mask(text_field_tensors).numpy(),
                            [[1, 1, 1, 0], [1, 1, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_list_field(self):
        text_field_tensors = {
                "list_tokens": torch.LongTensor([[[1, 2], [3, 0], [2, 0], [0, 0], [0, 0]],
                                                 [[5, 0], [4, 6], [0, 0], [0, 0], [0, 0]]])
        }
        actual_mask = util.get_text_field_mask(text_field_tensors, num_wrapping_dims=1).numpy()
        expected_mask = (text_field_tensors['list_tokens'].numpy() > 0).astype('int32')
        assert_almost_equal(actual_mask, expected_mask)

    def test_get_text_field_mask_returns_mask_key(self):
        text_field_tensors = {
                "tokens": torch.LongTensor([[3, 4, 5, 0, 0], [1, 2, 0, 0, 0]]),
                "mask": torch.LongTensor([[0, 0, 1]])
        }
        assert_almost_equal(util.get_text_field_mask(text_field_tensors).numpy(),
                            [[0, 0, 1]])

    def test_weighted_sum_works_on_simple_input(self):
        batch_size = 1
        sentence_length = 5
        embedding_dim = 4
        sentence_array = numpy.random.rand(batch_size, sentence_length, embedding_dim)
        sentence_tensor = torch.from_numpy(sentence_array).float()
        attention_tensor = torch.FloatTensor([[.3, .4, .1, 0, 1.2]])
        aggregated_array = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, embedding_dim)
        expected_array = (0.3 * sentence_array[0, 0] +
                          0.4 * sentence_array[0, 1] +
                          0.1 * sentence_array[0, 2] +
                          0.0 * sentence_array[0, 3] +
                          1.2 * sentence_array[0, 4])
        numpy.testing.assert_almost_equal(aggregated_array, [expected_array], decimal=5)

    def test_weighted_sum_handles_higher_order_input(self):
        batch_size = 1
        length_1 = 5
        length_2 = 6
        length_3 = 2
        embedding_dim = 4
        sentence_array = numpy.random.rand(batch_size, length_1, length_2, length_3, embedding_dim)
        attention_array = numpy.random.rand(batch_size, length_1, length_2, length_3)
        sentence_tensor = torch.from_numpy(sentence_array).float()
        attention_tensor = torch.from_numpy(attention_array).float()
        aggregated_array = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, length_2, embedding_dim)
        expected_array = (attention_array[0, 3, 2, 0] * sentence_array[0, 3, 2, 0] +
                          attention_array[0, 3, 2, 1] * sentence_array[0, 3, 2, 1])
        numpy.testing.assert_almost_equal(aggregated_array[0, 3, 2], expected_array, decimal=5)

    def test_weighted_sum_handles_uneven_higher_order_input(self):
        batch_size = 1
        length_1 = 5
        length_2 = 6
        length_3 = 2
        embedding_dim = 4
        sentence_array = numpy.random.rand(batch_size, length_3, embedding_dim)
        attention_array = numpy.random.rand(batch_size, length_1, length_2, length_3)
        sentence_tensor = torch.from_numpy(sentence_array).float()
        attention_tensor = torch.from_numpy(attention_array).float()
        aggregated_array = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, length_2, embedding_dim)
        for i in range(length_1):
            for j in range(length_2):
                expected_array = (attention_array[0, i, j, 0] * sentence_array[0, 0] +
                                  attention_array[0, i, j, 1] * sentence_array[0, 1])
                numpy.testing.assert_almost_equal(aggregated_array[0, i, j], expected_array,
                                                  decimal=5)

    def test_weighted_sum_handles_3d_attention_with_3d_matrix(self):
        batch_size = 1
        length_1 = 5
        length_2 = 2
        embedding_dim = 4
        sentence_array = numpy.random.rand(batch_size, length_2, embedding_dim)
        attention_array = numpy.random.rand(batch_size, length_1, length_2)
        sentence_tensor = torch.from_numpy(sentence_array).float()
        attention_tensor = torch.from_numpy(attention_array).float()
        aggregated_array = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, embedding_dim)
        for i in range(length_1):
            expected_array = (attention_array[0, i, 0] * sentence_array[0, 0] +
                              attention_array[0, i, 1] * sentence_array[0, 1])
            numpy.testing.assert_almost_equal(aggregated_array[0, i], expected_array,
                                              decimal=5)

    def test_viterbi_decode(self):
        # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
        sequence_logits = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
        transition_matrix = torch.zeros([9, 9])
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix)
        _, argmax_indices = torch.max(sequence_logits, 1)
        assert indices == argmax_indices.data.squeeze().tolist()

        # Test that pairwise potentials affect the sequence correctly and that
        # viterbi_decode can handle -inf values.
        sequence_logits = torch.FloatTensor([[0, 0, 0, 3, 5],
                                             [0, 0, 0, 3, 4],
                                             [0, 0, 0, 3, 4],
                                             [0, 0, 0, 3, 4],
                                             [0, 0, 0, 3, 4],
                                             [0, 0, 0, 3, 4]])
        # The same tags shouldn't appear sequentially.
        transition_matrix = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float("-inf")
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [4, 3, 4, 3, 4, 3]

        # Test that unbalanced pairwise potentials break ties
        # between paths with equal unary potentials.
        sequence_logits = torch.FloatTensor([[0, 0, 0, 4, 4],
                                             [0, 0, 0, 4, 4],
                                             [0, 0, 0, 4, 4],
                                             [0, 0, 0, 4, 4],
                                             [0, 0, 0, 4, 4],
                                             [0, 0, 0, 4, 4]])
        # The 5th tag has a penalty for appearing sequentially
        # or for transitioning to the 4th tag, making the best
        # path uniquely to take the 4th tag only.
        transition_matrix = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -10
        transition_matrix[3, 4] = -10
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 3, 3, 3, 3, 3]

        sequence_logits = torch.FloatTensor([[1, 0, 0, 4],
                                             [1, 0, 6, 2],
                                             [0, 3, 0, 4]])
        # Best path would normally be [3, 2, 3] but we add a
        # potential from 2 -> 1, making [3, 2, 1] the best path.
        transition_matrix = torch.zeros([4, 4])
        transition_matrix[0, 0] = 1
        transition_matrix[2, 1] = 5
        indices, value = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 2, 1]
        assert value.numpy() == 18

        # Test that providing evidence results in paths containing specified tags.
        sequence_logits = torch.FloatTensor([[0, 0, 0, 7, 7],
                                             [0, 0, 0, 7, 7],
                                             [0, 0, 0, 7, 7],
                                             [0, 0, 0, 7, 7],
                                             [0, 0, 0, 7, 7],
                                             [0, 0, 0, 7, 7]])
        # The 5th tag has a penalty for appearing sequentially
        # or for transitioning to the 4th tag, making the best
        # path to take the 4th tag for every label.
        transition_matrix = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -2
        transition_matrix[3, 4] = -2
        # The 1st, 4th and 5th sequence elements are observed - they should be
        # equal to 2, 0 and 4. The last tag should be equal to 3, because although
        # the penalty for transitioning to the 4th tag is -2, the unary potential
        # is 7, which is greater than the combination for any of the other labels.
        observations = [2, -1, -1, 0, 4, -1]
        indices, _ = util.viterbi_decode(sequence_logits,
                                         transition_matrix,
                                         observations)
        assert indices == [2, 3, 3, 0, 4, 3]

    def test_sequence_cross_entropy_with_logits_masks_loss_correctly(self):

        # test weight masking by checking that a tensor with non-zero values in
        # masked positions returns the same loss as a tensor with zeros in those
        # positions.
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        tensor2 = tensor.clone()
        tensor2[0, 3:, :] = 2
        tensor2[1, 4:, :] = 13
        tensor2[2, 2:, :] = 234
        tensor2[3, :, :] = 65
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights)
        loss2 = util.sequence_cross_entropy_with_logits(tensor2, targets, weights)
        assert loss.data.numpy() == loss2.data.numpy()

    def test_sequence_cross_entropy_with_logits_smooths_labels_correctly(self):
        tensor = torch.rand([1, 3, 4])
        targets = torch.LongTensor(numpy.random.randint(0, 3, [1, 3]))

        weights = torch.ones([2, 3])
        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, label_smoothing=0.1)

        correct_loss = 0.0
        for prediction, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            prediction = torch.nn.functional.log_softmax(prediction, dim=-1)
            correct_loss += prediction[label] * 0.9
            # incorrect elements
            correct_loss += prediction.sum() * 0.1 / 4
        # Average over sequence.
        correct_loss = - correct_loss / 3
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_averages_batch_correctly(self):
        # test batch average is the same as dividing the batch averaged
        # loss by the number of batches containing any non-padded tokens.
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights)

        vector_loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average=None)
        # Batch has one completely padded row, so divide by 4.
        assert loss.data.numpy() == vector_loss.sum().item() / 4

    def test_sequence_cross_entropy_with_logits_averages_token_correctly(self):
        # test token average is the same as multiplying the per-batch loss
        # with the per-batch weights and dividing by the total weight
        tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights

        loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average="token")

        vector_loss = util.sequence_cross_entropy_with_logits(tensor, targets, weights,
                                                              average=None)
        total_token_loss = (vector_loss * weights.float().sum(dim=-1)).sum()
        average_token_loss = (total_token_loss / weights.float().sum()).detach()
        assert_almost_equal(loss.detach().item(), average_token_loss.item())

    def test_replace_masked_values_replaces_masked_values_with_finite_value(self):
        tensor = torch.FloatTensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
        mask = torch.FloatTensor([[1, 1, 0]])
        replaced = util.replace_masked_values(tensor, mask.unsqueeze(-1), 2).data.numpy()
        assert_almost_equal(replaced, [[[1, 2, 3, 4], [5, 6, 7, 8], [2, 2, 2, 2]]])

    def test_logsumexp(self):
        # First a simple example where we add probabilities in log space.
        tensor = torch.FloatTensor([[.4, .1, .2]])
        log_tensor = tensor.log()
        log_summed = util.logsumexp(log_tensor, dim=-1, keepdim=False)
        assert_almost_equal(log_summed.exp().data.numpy(), [.7])
        log_summed = util.logsumexp(log_tensor, dim=-1, keepdim=True)
        assert_almost_equal(log_summed.exp().data.numpy(), [[.7]])

        # Then some more atypical examples, and making sure this will work with how we handle
        # log masks.
        tensor = torch.FloatTensor([[float('-inf'), 20.0]])
        assert_almost_equal(util.logsumexp(tensor).data.numpy(), [20.0])
        tensor = torch.FloatTensor([[-200.0, 20.0]])
        assert_almost_equal(util.logsumexp(tensor).data.numpy(), [20.0])
        tensor = torch.FloatTensor([[20.0, 20.0], [-200.0, 200.0]])
        assert_almost_equal(util.logsumexp(tensor, dim=0).data.numpy(), [20.0, 200.0])

    def test_flatten_and_batch_shift_indices(self):
        indices = numpy.array([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 9, 9, 9]],
                               [[2, 1, 0, 7],
                                [7, 7, 2, 3],
                                [0, 0, 4, 2]]])
        indices = torch.tensor(indices, dtype=torch.long)
        shifted_indices = util.flatten_and_batch_shift_indices(indices, 10)
        numpy.testing.assert_array_equal(shifted_indices.data.numpy(),
                                         numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                      9, 9, 9, 12, 11, 10, 17, 17,
                                                      17, 12, 13, 10, 10, 14, 12]))

    def test_batched_index_select(self):
        indices = numpy.array([[[1, 2],
                                [3, 4]],
                               [[5, 6],
                                [7, 8]]])
        # Each element is a vector of it's index.
        targets = torch.ones([2, 10, 3]).cumsum(1) - 1
        # Make the second batch double it's index so they're different.
        targets[1, :, :] *= 2
        indices = torch.tensor(indices, dtype=torch.long)
        selected = util.batched_index_select(targets, indices)

        assert list(selected.size()) == [2, 2, 2, 3]
        ones = numpy.ones([3])
        numpy.testing.assert_array_equal(selected[0, 0, 0, :].data.numpy(), ones)
        numpy.testing.assert_array_equal(selected[0, 0, 1, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[0, 1, 0, :].data.numpy(), ones * 3)
        numpy.testing.assert_array_equal(selected[0, 1, 1, :].data.numpy(), ones * 4)

        numpy.testing.assert_array_equal(selected[1, 0, 0, :].data.numpy(), ones * 10)
        numpy.testing.assert_array_equal(selected[1, 0, 1, :].data.numpy(), ones * 12)
        numpy.testing.assert_array_equal(selected[1, 1, 0, :].data.numpy(), ones * 14)
        numpy.testing.assert_array_equal(selected[1, 1, 1, :].data.numpy(), ones * 16)

    def test_flattened_index_select(self):
        indices = numpy.array([[1, 2],
                               [3, 4]])
        targets = torch.ones([2, 6, 3]).cumsum(1) - 1
        # Make the second batch double it's index so they're different.
        targets[1, :, :] *= 2
        indices = torch.tensor(indices, dtype=torch.long)

        selected = util.flattened_index_select(targets, indices)

        assert list(selected.size()) == [2, 2, 2, 3]

        ones = numpy.ones([3])
        numpy.testing.assert_array_equal(selected[0, 0, 0, :].data.numpy(), ones)
        numpy.testing.assert_array_equal(selected[0, 0, 1, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[0, 1, 0, :].data.numpy(), ones * 3)
        numpy.testing.assert_array_equal(selected[0, 1, 1, :].data.numpy(), ones * 4)

        numpy.testing.assert_array_equal(selected[1, 0, 0, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[1, 0, 1, :].data.numpy(), ones * 4)
        numpy.testing.assert_array_equal(selected[1, 1, 0, :].data.numpy(), ones * 6)
        numpy.testing.assert_array_equal(selected[1, 1, 1, :].data.numpy(), ones * 8)

        # Check we only accept 2D indices.
        with pytest.raises(ConfigurationError):
            util.flattened_index_select(targets, torch.ones([3, 4, 5]))

    def test_bucket_values(self):
        indices = torch.LongTensor([1, 2, 7, 1, 56, 900])
        bucketed_distances = util.bucket_values(indices)
        numpy.testing.assert_array_equal(bucketed_distances.numpy(),
                                         numpy.array([1, 2, 5, 1, 8, 9]))

    def test_add_sentence_boundary_token_ids_handles_2D_input(self):
        tensor = torch.from_numpy(numpy.array([[1, 2, 3], [4, 5, 0]]))
        mask = (tensor > 0).long()
        bos = 9
        eos = 10
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor = numpy.array([[9, 1, 2, 3, 10],
                                           [9, 4, 5, 10, 0]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == (expected_new_tensor > 0)).all()

    def test_add_sentence_boundary_token_ids_handles_3D_input(self):
        tensor = torch.from_numpy(
                numpy.array([[[1, 2, 3, 4],
                              [5, 5, 5, 5],
                              [6, 8, 1, 2]],
                             [[4, 3, 2, 1],
                              [8, 7, 6, 5],
                              [0, 0, 0, 0]]]))
        mask = ((tensor > 0).sum(dim=-1) > 0).type(torch.LongTensor)
        bos = torch.from_numpy(numpy.array([9, 9, 9, 9]))
        eos = torch.from_numpy(numpy.array([10, 10, 10, 10]))
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor = numpy.array([[[9, 9, 9, 9],
                                            [1, 2, 3, 4],
                                            [5, 5, 5, 5],
                                            [6, 8, 1, 2],
                                            [10, 10, 10, 10]],
                                           [[9, 9, 9, 9],
                                            [4, 3, 2, 1],
                                            [8, 7, 6, 5],
                                            [10, 10, 10, 10],
                                            [0, 0, 0, 0]]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == ((expected_new_tensor > 0).sum(axis=-1) > 0)).all()

    def test_remove_sentence_boundaries(self):
        tensor = torch.from_numpy(numpy.random.rand(3, 5, 7))
        mask = torch.from_numpy(
                # The mask with two elements is to test the corner case
                # of an empty sequence, so here we are removing boundaries
                # from  "<S> </S>"
                numpy.array([[1, 1, 0, 0, 0],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 0]])).long()
        new_tensor, new_mask = util.remove_sentence_boundaries(tensor, mask)

        expected_new_tensor = torch.zeros(3, 3, 7)
        expected_new_tensor[1, 0:3, :] = tensor[1, 1:4, :]
        expected_new_tensor[2, 0:2, :] = tensor[2, 1:3, :]
        assert_array_almost_equal(new_tensor.data.numpy(), expected_new_tensor.data.numpy())

        expected_new_mask = torch.from_numpy(
                numpy.array([[0, 0, 0],
                             [1, 1, 1],
                             [1, 1, 0]])).long()
        assert (new_mask.data.numpy() == expected_new_mask.data.numpy()).all()

    def test_add_positional_features(self):
        # This is hard to test, so we check that we get the same result as the
        # original tensorflow implementation:
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L270
        tensor2tensor_result = numpy.asarray([[0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00],
                                              [8.41470957e-01, 9.99999902e-05, 5.40302277e-01, 1.00000000e+00],
                                              [9.09297407e-01, 1.99999980e-04, -4.16146845e-01, 1.00000000e+00]])

        tensor = torch.zeros([2, 3, 4])
        result = util.add_positional_features(tensor, min_timescale=1.0, max_timescale=1.0e4)
        numpy.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        numpy.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)

        # Check case with odd number of dimensions.
        tensor2tensor_result = numpy.asarray([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
                                               1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                              [8.41470957e-01, 9.99983307e-03, 9.99999902e-05, 5.40302277e-01,
                                               9.99949992e-01, 1.00000000e+00, 0.00000000e+00],
                                              [9.09297407e-01, 1.99986659e-02, 1.99999980e-04, -4.16146815e-01,
                                               9.99800026e-01, 1.00000000e+00, 0.00000000e+00]])

        tensor = torch.zeros([2, 3, 7])
        result = util.add_positional_features(tensor, min_timescale=1.0, max_timescale=1.0e4)
        numpy.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        numpy.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)

    def test_combine_tensors_and_multiply(self):
        tensors = [torch.Tensor([[[2, 3]]]), torch.Tensor([[[5, 5]]])]
        weight = torch.Tensor([4, 5])

        combination = "x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[8 + 15]])

        combination = "y"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[20 + 25]])

        combination = "x,y"
        weight2 = torch.Tensor([4, 5, 4, 5])
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight2),
                            [[8 + 20 + 15 + 25]])

        combination = "x-y"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[-3 * 4 + -2 * 5]])

        combination = "y-x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[3 * 4 + 2 * 5]])

        combination = "y+x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[7 * 4 + 8 * 5]])

        combination = "y*x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[10 * 4 + 15 * 5]])

        combination = "y/x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[(5 / 2) * 4 + (5 / 3) * 5]], decimal=4)

        combination = "x/y"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[(2 / 5) * 4 + (3 / 5) * 5]], decimal=4)

        with pytest.raises(ConfigurationError):
            util.combine_tensors_and_multiply("x+y+y", tensors, weight)

        with pytest.raises(ConfigurationError):
            util.combine_tensors_and_multiply("x%y", tensors, weight)

    def test_combine_tensors_and_multiply_with_same_batch_size_and_embedding_dim(self):
        # This test just makes sure we handle some potential edge cases where the lengths of all
        # dimensions are the same, making sure that the multiplication with the weight vector
        # happens along the right dimension (it should be the last one).
        tensors = [torch.Tensor([[[5, 5], [4, 4]], [[2, 3], [1, 1]]])]  # (2, 2, 2)
        weight = torch.Tensor([4, 5])  # (2,)

        combination = "x"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[20 + 25, 16 + 20], [8 + 15, 4 + 5]])

        tensors = [torch.Tensor([[[5, 5], [2, 2]], [[4, 4], [3, 3]]]),
                   torch.Tensor([[[2, 3]], [[1, 1]]])]
        weight = torch.Tensor([4, 5])
        combination = "x*y"
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight),
                            [[5 * 2 * 4 + 5 * 3 * 5, 2 * 2 * 4 + 2 * 3 * 5],
                             [4 * 1 * 4 + 4 * 1 * 5, 3 * 1 * 4 + 3 * 1 * 5]])

    def test_combine_tensors_and_multiply_with_batch_size_one(self):
        seq_len_1 = 10
        seq_len_2 = 5
        embedding_dim = 8

        combination = "x,y,x*y"
        t1 = torch.randn(1, seq_len_1, embedding_dim)
        t2 = torch.randn(1, seq_len_2, embedding_dim)
        combined_dim = util.get_combined_dim(combination, [embedding_dim, embedding_dim])
        weight = torch.Tensor(combined_dim)

        result = util.combine_tensors_and_multiply(combination, [t1.unsqueeze(2), t2.unsqueeze(1)], weight)

        assert_almost_equal(
                result.size(),
                [1, seq_len_1, seq_len_2]
        )

    def test_combine_tensors_and_multiply_with_batch_size_one_and_seq_len_one(self):
        seq_len_1 = 10
        seq_len_2 = 1
        embedding_dim = 8

        combination = "x,y,x*y"
        t1 = torch.randn(1, seq_len_1, embedding_dim)
        t2 = torch.randn(1, seq_len_2, embedding_dim)
        combined_dim = util.get_combined_dim(combination, [embedding_dim, embedding_dim])
        weight = torch.Tensor(combined_dim)

        result = util.combine_tensors_and_multiply(combination, [t1.unsqueeze(2), t2.unsqueeze(1)], weight)

        assert_almost_equal(
                result.size(),
                [1, seq_len_1, seq_len_2]
        )

    def test_has_tensor(self):
        # pylint: disable=bad-continuation
        has_tensor = util.has_tensor
        tensor = torch.tensor([1, 2, 3])

        assert has_tensor(["a", 10, tensor])
        assert not has_tensor(["a", 10])

        assert has_tensor(("a", 10, tensor))
        assert not has_tensor(("a", 10))

        assert has_tensor({"a": tensor, "b": 1})
        assert not has_tensor({"a": 10, "b": 1})

        assert has_tensor(tensor)
        assert not has_tensor(3)

        assert has_tensor({
                "x": [
                        0,
                        {
                                "inside": {
                                        "double_inside": [
                                                3,
                                                [
                                                        10,
                                                        tensor
                                                ]
                                        ]
                                }
                        }
                ]
        })

    def test_combine_initial_dims(self):
        tensor = torch.randn(4, 10, 20, 17, 5)

        tensor2d = util.combine_initial_dims(tensor)
        assert list(tensor2d.size()) == [4 * 10 * 20 * 17, 5]

    def test_uncombine_initial_dims(self):
        embedding2d = torch.randn(4 * 10 * 20 * 17 * 5, 12)

        embedding = util.uncombine_initial_dims(embedding2d, torch.Size((4, 10, 20, 17, 5)))
        assert list(embedding.size()) == [4, 10, 20, 17, 5, 12]
