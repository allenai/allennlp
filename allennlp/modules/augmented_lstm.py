"""
An LSTM with Recurrent Dropout and the option to use highway
connections between layers.
Based on PyText version (that was based on a previous AllenNLP version)
"""

from typing import Optional, Tuple

import torch
from allennlp.common.checks import ConfigurationError
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from allennlp.nn.initializers import block_orthogonal
from allennlp.nn.util import get_dropout_mask


class AugmentedLSTMCell(torch.nn.Module):
    """
    `AugmentedLSTMCell` implements a AugmentedLSTM cell.

    # Parameters

    embed_dim : `int`
        The number of expected features in the input.
    lstm_dim : `int`
        Number of features in the hidden state of the LSTM.
    use_highway : `bool`, optional (default = `True`)
        If `True` we append a highway network to the outputs of the LSTM.
    use_bias : `bool`, optional (default = `True`)
        If `True` we use a bias in our LSTM calculations, otherwise we don't.

    # Attributes

    input_linearity : `nn.Module`
        Fused weight matrix which computes a linear function over the input.
    state_linearity : `nn.Module`
        Fused weight matrix which computes a linear function over the states.
    """

    def __init__(
        self, embed_dim: int, lstm_dim: int, use_highway: bool = True, use_bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.use_highway = use_highway
        self.use_bias = use_bias

        if use_highway:
            self._highway_inp_proj_start = 5 * self.lstm_dim
            self._highway_inp_proj_end = 6 * self.lstm_dim

            # fused linearity of input to input_gate,
            # forget_gate, memory_init, output_gate, highway_gate,
            # and the actual highway value
            self.input_linearity = torch.nn.Linear(
                self.embed_dim, self._highway_inp_proj_end, bias=self.use_bias
            )
            # fused linearity of input to input_gate,
            # forget_gate, memory_init, output_gate, highway_gate
            self.state_linearity = torch.nn.Linear(
                self.lstm_dim, self._highway_inp_proj_start, bias=True
            )
        else:
            # If there's no highway layer then we have a standard
            # LSTM. The 4 comes from fusing input, forget, memory, output
            # gates/inputs.
            self.input_linearity = torch.nn.Linear(
                self.embed_dim, 4 * self.lstm_dim, bias=self.use_bias
            )
            self.state_linearity = torch.nn.Linear(self.lstm_dim, 4 * self.lstm_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.lstm_dim, self.embed_dim])
        block_orthogonal(self.state_linearity.weight.data, [self.lstm_dim, self.lstm_dim])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.lstm_dim : 2 * self.lstm_dim].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        states=Tuple[torch.Tensor, torch.Tensor],
        variational_dropout_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        !!! Warning
            DO NOT USE THIS LAYER DIRECTLY, instead use the AugmentedLSTM class

        # Parameters

        x : `torch.Tensor`
            Input tensor of shape (bsize x input_dim).
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing
            the hidden state and the cell state of each element in
            the batch. Each of these tensors have a dimension of
            (bsize x nhid). Defaults to `None`.

        # Returns

        `Tuple[torch.Tensor, torch.Tensor]`
            Returned states. Shape of each state is (bsize x nhid).

        """
        hidden_state, memory_state = states

        # In Pytext this was done as the last step of the cell.
        # But the original AugmentedLSTM from AllenNLP this was done before the processing
        if variational_dropout_mask is not None and self.training:
            hidden_state = hidden_state * variational_dropout_mask

        projected_input = self.input_linearity(x)
        projected_state = self.state_linearity(hidden_state)

        input_gate = forget_gate = memory_init = output_gate = highway_gate = None
        if self.use_highway:
            fused_op = projected_input[:, : 5 * self.lstm_dim] + projected_state
            fused_chunked = torch.chunk(fused_op, 5, 1)
            (input_gate, forget_gate, memory_init, output_gate, highway_gate) = fused_chunked
            highway_gate = torch.sigmoid(highway_gate)
        else:
            fused_op = projected_input + projected_state
            input_gate, forget_gate, memory_init, output_gate = torch.chunk(fused_op, 4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        memory_init = torch.tanh(memory_init)
        output_gate = torch.sigmoid(output_gate)
        memory = input_gate * memory_init + forget_gate * memory_state
        timestep_output: torch.Tensor = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_input_projection = projected_input[
                :, self._highway_inp_proj_start : self._highway_inp_proj_end
            ]
            timestep_output = (
                highway_gate * timestep_output
                + (1 - highway_gate) * highway_input_projection  # noqa
            )

        return timestep_output, memory


class AugmentedLstm(torch.nn.Module):
    """
    `AugmentedLstm` implements a one-layer single directional
    AugmentedLSTM layer. AugmentedLSTM is an LSTM which optionally
    appends an optional highway network to the output layer. Furthermore the
    dropout controls the level of variational dropout done.

    # Parameters

    input_size : `int`
        The number of expected features in the input.
    hidden_size : `int`
        Number of features in the hidden state of the LSTM.
        Defaults to 32.
    go_forward : `bool`
        Whether to compute features left to right (forward)
        or right to left (backward).
    recurrent_dropout_probability : `float`
        Variational dropout probability to use. Defaults to 0.0.
    use_highway : `bool`
        If `True` we append a highway network to the outputs of the LSTM.
    use_input_projection_bias : `bool`
        If `True` we use a bias in our LSTM calculations, otherwise we don't.

    # Attributes

    cell : `AugmentedLSTMCell`
        `AugmentedLSTMCell` that is applied at every timestep.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        go_forward: bool = True,
        recurrent_dropout_probability: float = 0.0,
        use_highway: bool = True,
        use_input_projection_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = input_size
        self.lstm_dim = hidden_size

        self.go_forward = go_forward
        self.use_highway = use_highway
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.cell = AugmentedLSTMCell(
            self.embed_dim, self.lstm_dim, self.use_highway, use_input_projection_bias
        )

    def forward(
        self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Warning: Would be better to use the BiAugmentedLstm class in a regular model

        Given an input batch of sequential data such as word embeddings, produces a single layer unidirectional
        AugmentedLSTM representation of the sequential input and new state tensors.

        # Parameters

        inputs : `PackedSequence`
            `bsize` sequences of shape `(len, input_dim)` each, in PackedSequence format
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing the initial hidden state and
            the cell state of each element in the batch. Each of these tensors have a dimension of
            (1 x bsize x nhid). Defaults to `None`.

        # Returns

        `Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]`
            AugmentedLSTM representation of input and the state of the LSTM `t = seq_len`.
            Shape of representation is (bsize x seq_len x representation_dim).
            Shape of each state is (1 x bsize x nhid).

        """
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError("inputs must be PackedSequence but got %s" % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]
        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.lstm_dim)
        if states is None:
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.lstm_dim)
            full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size, self.lstm_dim)
        else:
            full_batch_previous_state = states[0].squeeze(0)
            full_batch_previous_memory = states[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(
                self.recurrent_dropout_probability, full_batch_previous_memory
            )
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum
                # number of elements in the batch?
                # Second conditional: Does the next shortest
                # sequence beyond the current batch
                # index require computation use this timestep?
                while (
                    current_length_index < (len(batch_lengths) - 1)
                    and batch_lengths[current_length_index + 1] > index
                ):
                    current_length_index += 1

            previous_memory = full_batch_previous_memory[0 : current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0 : current_length_index + 1].clone()
            timestep_input = sequence_tensor[0 : current_length_index + 1, index]
            timestep_output, memory = self.cell(
                timestep_input,
                (previous_state, previous_memory),
                dropout_mask[0 : current_length_index + 1] if dropout_mask is not None else None,
            )
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0 : current_length_index + 1] = memory
            full_batch_previous_state[0 : current_length_index + 1] = timestep_output
            output_accumulator[0 : current_length_index + 1, index, :] = timestep_output

        output_accumulator = pack_padded_sequence(
            output_accumulator, batch_lengths, batch_first=True
        )

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, lstm_dim). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (
            full_batch_previous_state.unsqueeze(0),
            full_batch_previous_memory.unsqueeze(0),
        )
        return output_accumulator, final_state


class BiAugmentedLstm(torch.nn.Module):
    """
    `BiAugmentedLstm` implements a generic AugmentedLSTM representation layer.
    BiAugmentedLstm is an LSTM which optionally appends an optional highway network to the output layer.
    Furthermore the dropout controls the level of variational dropout done.

    # Parameters

    input_size : `int`, required
        The dimension of the inputs to the LSTM.
    hidden_size : `int`, required.
        The dimension of the outputs of the LSTM.
    num_layers : `int`
        Number of recurrent layers. Eg. setting `num_layers=2`
        would mean stacking two LSTMs together to form a stacked LSTM,
        with the second LSTM taking in the outputs of the first LSTM and
        computing the final result. Defaults to 1.
    bias : `bool`
        If `True` we use a bias in our LSTM calculations, otherwise we don't.
    recurrent_dropout_probability : `float`, optional (default = `0.0`)
        Variational dropout probability to use.
    bidirectional : `bool`
        If `True`, becomes a bidirectional LSTM. Defaults to `True`.
    padding_value : `float`, optional (default = `0.0`)
        Value for the padded elements. Defaults to 0.0.
    use_highway : `bool`, optional (default = `True`)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)

    # Returns

    output_accumulator : `PackedSequence`
        The outputs of the LSTM for each timestep. A tensor of shape (batch_size, max_timesteps, hidden_size) where
        for a given batch element, all outputs past the sequence length for that batch are zero tensors.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        recurrent_dropout_probability: float = 0.0,
        bidirectional: bool = False,
        padding_value: float = 0.0,
        use_highway: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.padding_value = padding_value
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.use_highway = use_highway
        self.use_bias = bias

        num_directions = int(self.bidirectional) + 1
        self.forward_layers = torch.nn.ModuleList()
        if self.bidirectional:
            self.backward_layers = torch.nn.ModuleList()

        lstm_embed_dim = self.input_size
        for _ in range(self.num_layers):
            self.forward_layers.append(
                AugmentedLstm(
                    lstm_embed_dim,
                    self.hidden_size,
                    go_forward=True,
                    recurrent_dropout_probability=self.recurrent_dropout_probability,
                    use_highway=self.use_highway,
                    use_input_projection_bias=self.use_bias,
                )
            )
            if self.bidirectional:
                self.backward_layers.append(
                    AugmentedLstm(
                        lstm_embed_dim,
                        self.hidden_size,
                        go_forward=False,
                        recurrent_dropout_probability=self.recurrent_dropout_probability,
                        use_highway=self.use_highway,
                        use_input_projection_bias=self.use_bias,
                    )
                )

            lstm_embed_dim = self.hidden_size * num_directions
        self.representation_dim = lstm_embed_dim

    def forward(
        self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given an input batch of sequential data such as word embeddings, produces
        a AugmentedLSTM representation of the sequential input and new state
        tensors.

        # Parameters

        inputs : `PackedSequence`, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        states : `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of tensors containing
            the initial hidden state and the cell state of each element in
            the batch. Each of these tensors have a dimension of
            (bsize x num_layers x num_directions * nhid). Defaults to `None`.

        # Returns

        `Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
            AgumentedLSTM representation of input and
            the state of the LSTM `t = seq_len`.
            Shape of representation is (bsize x seq_len x representation_dim).
            Shape of each state is (bsize x num_layers * num_directions x nhid).

        """

        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError("inputs must be PackedSequence but got %s" % (type(inputs)))

        # if states is not None:
        #    states = (states[0].transpose(0, 1), states[1].transpose(0, 1))
        if self.bidirectional:
            return self._forward_bidirectional(inputs, states)
        return self._forward_unidirectional(inputs, states)

    def _forward_bidirectional(
        self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ):
        output_sequence = inputs
        final_h = []
        final_c = []

        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(
                zip(  # type: ignore
                    states[0].chunk(self.num_layers, 0), states[1].chunk(self.num_layers, 0)
                )
            )
        for i, state in enumerate(hidden_states):
            if state:
                forward_state = state[0].chunk(2, -1)
                backward_state = state[1].chunk(2, -1)
            else:
                forward_state = backward_state = None

            forward_layer = self.forward_layers[i]
            backward_layer = self.backward_layers[i]
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, forward_state)
            backward_output, final_backward_state = backward_layer(output_sequence, backward_state)
            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)
            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)

            final_h.extend([final_forward_state[0], final_backward_state[0]])
            final_c.extend([final_forward_state[1], final_backward_state[1]])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)
        output_sequence, batch_lengths = pad_packed_sequence(
            output_sequence, padding_value=self.padding_value, batch_first=True
        )

        output_sequence = pack_padded_sequence(output_sequence, batch_lengths, batch_first=True)
        return output_sequence, final_state_tuple

    def _forward_unidirectional(
        self, inputs: PackedSequence, states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ):
        output_sequence = inputs
        final_h = []
        final_c = []

        if not states:
            hidden_states = [None] * self.num_layers
        elif states[0].size()[0] != self.num_layers:
            raise RuntimeError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(
                zip(  # type: ignore
                    states[0].chunk(self.num_layers, 0), states[1].chunk(self.num_layers, 0)
                )  # type: ignore
            )

        for i, state in enumerate(hidden_states):
            forward_layer = self.forward_layers[i]
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            output_sequence = forward_output
            final_h.append(final_forward_state[0])
            final_c.append(final_forward_state[1])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)
        output_sequence, batch_lengths = pad_packed_sequence(
            output_sequence, padding_value=self.padding_value, batch_first=True
        )

        output_sequence = pack_padded_sequence(output_sequence, batch_lengths, batch_first=True)

        return output_sequence, final_state_tuple
