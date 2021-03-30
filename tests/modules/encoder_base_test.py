import numpy
import pytest
import torch
from torch.nn import LSTM, RNN

from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


class TestEncoderBase(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
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
        mask = torch.ones(5, 7).bool()
        mask[1, 6:] = False
        mask[2, :] = False  # <= completely masked
        mask[3, 2:] = False
        mask[4, :] = False  # <= completely masked
        self.mask = mask

        self.batch_size = 5
        self.num_valid = 3
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        _, _, restoration_indices, sorting_indices = sort_batch_by_length(tensor, sequence_lengths)
        self.sorting_indices = sorting_indices
        self.restoration_indices = restoration_indices

    def test_non_stateful_states_are_sorted_correctly(self):
        encoder_base = _EncoderBase(stateful=False)
        initial_states = (torch.randn(6, 5, 7), torch.randn(6, 5, 7))
        # Check that we sort the state for non-stateful encoders. To test
        # we'll just use a "pass through" encoder, as we aren't actually testing
        # the functionality of the encoder here anyway.
        _, states, restoration_indices = encoder_base.sort_and_run_forward(
            lambda *x: x, self.tensor, self.mask, initial_states
        )
        # Our input tensor had 2 zero length sequences, so we need
        # to concat a tensor of shape
        # (num_layers * num_directions, batch_size - num_valid, hidden_dim),
        # to the output before unsorting it.
        zeros = torch.zeros([6, 2, 7])

        # sort_and_run_forward strips fully-padded instances from the batch;
        # in order to use the restoration_indices we need to add back the two
        #  that got stripped. What we get back should match what we started with.
        for state, original in zip(states, initial_states):
            assert list(state.size()) == [6, 3, 7]
            state_with_zeros = torch.cat([state, zeros], 1)
            unsorted_state = state_with_zeros.index_select(1, restoration_indices)
            for index in [0, 1, 3]:
                numpy.testing.assert_array_equal(
                    unsorted_state[:, index, :].data.numpy(), original[:, index, :].data.numpy()
                )

    def test_get_initial_states(self):
        # First time we call it, there should be no state, so we should return None.
        assert (
            self.encoder_base._get_initial_states(
                self.batch_size, self.num_valid, self.sorting_indices
            )
            is None
        )

        # First test the case that the previous state is _smaller_ than the current state input.
        initial_states = (torch.randn([1, 3, 7]), torch.randn([1, 3, 7]))
        self.encoder_base._states = initial_states
        # sorting indices are: [0, 1, 3, 2, 4]
        returned_states = self.encoder_base._get_initial_states(
            self.batch_size, self.num_valid, self.sorting_indices
        )

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

        # The returned states should be of shape (1, num_valid, hidden_size) and
        # they also should have been sorted with respect to the indices.
        # sorting indices are: [0, 1, 3, 2, 4]

        correct_returned_states = [
            state.index_select(1, self.sorting_indices)[:, : self.num_valid, :]
            for state in correct_expanded_states
        ]

        numpy.testing.assert_array_equal(
            returned_states[0].data.numpy(), correct_returned_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            returned_states[1].data.numpy(), correct_returned_states[1].data.numpy()
        )

        # Now test the case that the previous state is larger:
        original_states = (torch.randn([1, 10, 7]), torch.randn([1, 10, 7]))
        self.encoder_base._states = original_states
        # sorting indices are: [0, 1, 3, 2, 4]
        returned_states = self.encoder_base._get_initial_states(
            self.batch_size, self.num_valid, self.sorting_indices
        )
        # State should not have changed, as they were larger
        # than the batch size of the requested states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(), original_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(), original_states[1].data.numpy()
        )

        # The returned states should be of shape (1, num_valid, hidden_size) and they
        # also should have been sorted with respect to the indices.
        correct_returned_state = [
            x.index_select(1, self.sorting_indices)[:, : self.num_valid, :] for x in original_states
        ]
        numpy.testing.assert_array_equal(
            returned_states[0].data.numpy(), correct_returned_state[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            returned_states[1].data.numpy(), correct_returned_state[1].data.numpy()
        )

    def test_update_states(self):
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])

        index_selected_initial_states = (
            initial_states[0].index_select(1, self.restoration_indices),
            initial_states[1].index_select(1, self.restoration_indices),
        )

        self.encoder_base._update_states(initial_states, self.restoration_indices)
        # State was None, so the updated state should just be the sorted given state.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0].data.numpy(), index_selected_initial_states[0].data.numpy()
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1].data.numpy(), index_selected_initial_states[1].data.numpy()
        )

        new_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        # tensor has 2 completely masked rows, so the last 2 rows of the _sorted_ states
        # will be completely zero, having been appended after calling the respective encoder.
        new_states[0][:, -2:, :] = 0
        new_states[1][:, -2:, :] = 0

        index_selected_new_states = (
            new_states[0].index_select(1, self.restoration_indices),
            new_states[1].index_select(1, self.restoration_indices),
        )

        self.encoder_base._update_states(new_states, self.restoration_indices)
        # Check that the update _preserved_ the state for the rows which were
        # completely masked (2 and 4):
        for index in [2, 4]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_initial_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_initial_states[1][:, index, :].data.numpy(),
            )
        # Now the states which were updated:
        for index in [0, 1, 3]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_new_states[1][:, index, :].data.numpy(),
            )

        # Now test the case that the new state is smaller:
        small_new_states = torch.randn([1, 3, 7]), torch.randn([1, 3, 7])
        # pretend the 2nd sequence in the batch was fully masked.
        small_restoration_indices = torch.LongTensor([2, 0, 1])
        small_new_states[0][:, 0, :] = 0
        small_new_states[1][:, 0, :] = 0

        index_selected_small_states = (
            small_new_states[0].index_select(1, small_restoration_indices),
            small_new_states[1].index_select(1, small_restoration_indices),
        )
        self.encoder_base._update_states(small_new_states, small_restoration_indices)

        # Check the index for the row we didn't update is the same as the previous step:
        for index in [1, 3]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_new_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_new_states[1][:, index, :].data.numpy(),
            )
        # Indices we did update:
        for index in [0, 2]:
            numpy.testing.assert_array_equal(
                self.encoder_base._states[0][:, index, :].data.numpy(),
                index_selected_small_states[0][:, index, :].data.numpy(),
            )
            numpy.testing.assert_array_equal(
                self.encoder_base._states[1][:, index, :].data.numpy(),
                index_selected_small_states[1][:, index, :].data.numpy(),
            )

        # We didn't update index 4 in the previous step either, so it should be equal to the
        # 4th index of initial states.
        numpy.testing.assert_array_equal(
            self.encoder_base._states[0][:, 4, :].data.numpy(),
            index_selected_initial_states[0][:, 4, :].data.numpy(),
        )
        numpy.testing.assert_array_equal(
            self.encoder_base._states[1][:, 4, :].data.numpy(),
            index_selected_initial_states[1][:, 4, :].data.numpy(),
        )

    def test_reset_states(self):
        # Initialize the encoder states.
        assert self.encoder_base._states is None
        initial_states = torch.randn([1, 5, 7]), torch.randn([1, 5, 7])
        index_selected_initial_states = (
            initial_states[0].index_select(1, self.restoration_indices),
            initial_states[1].index_select(1, self.restoration_indices),
        )
        self.encoder_base._update_states(initial_states, self.restoration_indices)

        # Check that only some of the states are reset when a mask is provided.
        mask = torch.tensor([True, True, False, False, False])
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
        bad_mask = torch.tensor([True, True, False])
        with pytest.raises(ValueError):
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
        encoder_base._update_states(final_states, self.restoration_indices)
        encoder_base.sort_and_run_forward(self.lstm, self.tensor, self.mask)
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]], self.restoration_indices)
        encoder_base.sort_and_run_forward(self.rnn, self.tensor, self.mask)

    @requires_gpu
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
        encoder_base._update_states(final_states, self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.lstm.cuda(), self.tensor.cuda(), self.mask.cuda())
        # Check RNN
        encoder_base.reset_states()
        encoder_base._update_states([final_states[0]], self.restoration_indices.cuda())
        encoder_base.sort_and_run_forward(self.rnn.cuda(), self.tensor.cuda(), self.mask.cuda())
