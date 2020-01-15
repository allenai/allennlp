import numpy
import pytest
import torch
from torch.nn import LSTM, RNN

from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


class TestEncoderBase(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.lstm = LSTM(
            bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True
        )
        self.rnn = RNN(
            bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True
        )
        self.encoder_base = _EncoderBase(stateful=True)

        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[3, 2:, :] = 0
        self.tensor = tensor
        mask = torch.ones(5, 7)
        mask[1, 6:] = 0
        mask[2, :] = 0  # <= completely masked
        mask[3, 2:] = 0
        mask[4, :] = 0  # <= completely masked
        self.mask = mask

        self.batch_size = 5
        self.num_valid = 3

    def test_non_stateful_states_are_sorted_correctly(self):
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (torch.randn(6, 5, 7), torch.randn(6, 5, 7))
        # Check that we sort the state for non-stateful encoders. To test
        # we'll just use a "pass through" encoder, as we aren't actually testing
        # the functionality of the encoder here anyway.
        _, states = encoder_base.sort_and_run_forward(
            lambda *x: x, self.tensor, self.mask, initial_states
        )
        # What we get back should match what we started with.
        for state, original in zip(states, initial_states):
            assert list(state.size()) == [6, 5, 7]
            for index in [0, 1, 3]:
                numpy.testing.assert_array_equal(
                    state[:, index, :].data.numpy(), original[:, index, :].data.numpy()
                )

    def test_get_initial_states(self):
        # First time we call it, there should be no state, so we should return None.
        assert self.encoder_base._get_initial_states(self.batch_size) is None

        # First test the case that the previous state is _smaller_ than the current state input.
        initial_states = (torch.randn([1, 3, 7]), torch.randn([1, 3, 7]))
        self.encoder_base._states = initial_states
        # sorting indices are: [0, 1, 3, 2, 4]
        _ = self.encoder_base._get_initial_states(self.batch_size)

        correct_expanded_states = [
            torch.cat([state, torch.zeros([1, 2, 7])], 1) for state in initial_states
        ]
        # State should have been expanded with zeros to have shape (1, batch_size, hidden_size).
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(), correct_expanded_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(), correct_expanded_states[1].data.numpy()
        )

        # Now test the case that the previous state is larger:
        original_states = (torch.randn([1, 10, 7]), torch.randn([1, 10, 7]))
        self.encoder_base._states = original_states
        # sorting indices are: [0, 1, 3, 2, 4]
        _ = self.encoder_base._get_initial_states(self.batch_size)
        # State should not have changed, as they were larger
        # than the batch size of the requested states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(), original_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(), original_states[1].data.numpy()
        )

    def test_update_states(self):
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])

        self.encoder_base._update_states(initial_states)
        # State was None, so the updated state should just be the sorted given state.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(), initial_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(), initial_states[1].data.numpy()
        )

        new_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        # tensor has 2 completely masked rows, so the last 2 rows of the states
        # will be completely zero, having been appended after calling the respective encoder.
        new_states[0][:, -2:, :] = 0
        new_states[1][:, -2:, :] = 0

        self.encoder_base._update_states(new_states)
        # Check that the update _preserved_ the state for the rows which were
        # completely masked (3 and 4):
        for index in [3, 4]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                initial_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                initial_states[1][:, index, :].data.numpy(),
            )
        # Now the states which were updated:
        for index in [0, 1, 2]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                new_states[1][:, index, :].data.numpy(),
            )

        # Now test the case that the new state is smaller:
        small_new_states = torch.randn([1, 3, 7]), torch.randn([1, 3, 7])
        # pretend the 1st sequence in the batch was fully masked.
        small_new_states[0][:, 0, :] = 0
        small_new_states[1][:, 0, :] = 0

        self.encoder_base._update_states(small_new_states)

        # Check the index for the row we didn't update is the same as the previous step:
        for index in [0, 3]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                new_states[1][:, index, :].data.numpy(),
            )
        # Indices we did update:
        for index in [1, 2]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                small_new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                small_new_states[1][:, index, :].data.numpy(),
            )

        # We didn't update index 4 in the previous step either, so it should be equal to the
        # 4th index of initial states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, 4, :].data.numpy(),
            initial_states[0][:, 4, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, 4, :].data.numpy(),
            initial_states[1][:, 4, :].data.numpy(),
        )

    def test_reset_states(self):
        # Initialize the encoder states.
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        index_selected_initial_states = (
            initial_states[0],
            initial_states[1],
        )
        self.encoder_base._update_states(initial_states)

        # Check that only some of the states are reset when a mask is provided.
        mask = torch.FloatTensor([1, 1, 0, 0, 0])
        self.encoder_base.reset_states(mask)
        # First two states should be zeros
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, :2, :].data.numpy(),
            torch.zeros_like(initial_states[0])[:, :2, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, :2, :].data.numpy(),
            torch.zeros_like(initial_states[1])[:, :2, :].data.numpy(),
        )
        # Remaining states should be the same
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, 2:, :].data.numpy(),
            index_selected_initial_states[0][:, 2:, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, 2:, :].data.numpy(),
            index_selected_initial_states[1][:, 2:, :].data.numpy(),
        )

        # Check that error is raised if mask has wrong batch size.
        bad_mask = torch.FloatTensor([1, 1, 0])
        with self.assertRaises(ValueError):
            self.encoder_base.reset_states(bad_mask)

        # Check that states are reset to None if no mask is provided.
        self.encoder_base.reset_states()
        assert self.encoder_base._states is None

    def test_non_contiguous_initial_states_handled(self):
        # Check that the encoder is robust to non-contiguous initial states.

        # Case 1: Encoder is not stateful

        # A transposition will make the tensors non-contiguous, start them off at the wrong shape
        # and transpose them into the right shape.
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (
            torch.randn(5, 6, 7).permute(1, 0, 2),
            torch.randn(5, 6, 7).permute(1, 0, 2),
        )
        assert not initial_states[0].is_contiguous() and not initial_states[1].is_contiguous()
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])

        # We'll pass them through an LSTM encoder and a vanilla RNN encoder to make sure it works
        # whether the initial states are a tuple of tensors or just a single tensor.
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask, initial_states)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask, initial_states[0])

        # Case 2: Encoder is stateful

        # For stateful encoders, the initial state may be non-contiguous if its state was
        # previously updated with non-contiguous tensors. As in the non-stateful tests, we check
        # that the encoder still works on initial states for RNNs and LSTMs.
        final_states = initial_states
        # Check LSTM
        encoder_base = _EncoderBase(stateful=True)
        encoder_base._update_states(final_states)
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask)
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]])
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_non_contiguous_initial_states_handled_on_gpu(self):
        # Some PyTorch operations which produce contiguous tensors on the CPU produce
        # non-contiguous tensors on the GPU (e.g. forward pass of an RNN when batch_first=True).
        # Accordingly, we perform the same checks from previous test on the GPU to ensure the
        # encoder is not affected by which device it is on.

        # Case 1: Encoder is not stateful

        # A transposition will make the tensors non-contiguous, start them off at the wrong shape
        # and transpose them into the right shape.
        encoder_base = _EncoderBase(stateful=False).cuda()
        initial_states = (
            torch.randn(5, 6, 7).cuda().permute(1, 0, 2),
            torch.randn(5, 6, 7).cuda().permute(1, 0, 2),
        )
        assert not initial_states[0].is_contiguous() and not initial_states[1].is_contiguous()
        assert initial_states[0].size() == torch.Size([6, 5, 7])
        assert initial_states[1].size() == torch.Size([6, 5, 7])

        # We'll pass them through an LSTM encoder and a vanilla RNN encoder to make sure it works
        # whether the initial states are a tuple of tensors or just a single tensor.
        encoder_base.sort_and_run_forward(
            self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda(), initial_states
        )
        encoder_base.sort_and_run_forward(
            self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda(), initial_states[0]
        )

        # Case 2: Encoder is stateful

        # For stateful encoders, the initial state may be non-contiguous if its state was
        # previously updated with non-contiguous tensors. As in the non-stateful tests, we check
        # that the encoder still works on initial states for RNNs and LSTMs.
        final_states = initial_states
        # Check LSTM
        encoder_base = _EncoderBase(stateful=True).cuda()
        encoder_base._update_states(final_states)
        encoder_base.sort_and_run_forward(self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda())
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]])
        encoder_base.sort_and_run_forward(self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda())
