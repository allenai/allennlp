
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.nn.modules import Dropout


class AugmentedLstm(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    output_size : int, required
        The dimension of the outputs of the LSTM.
    direction: str, optional (default = "forward")
        The direction in which the LSTM is applied to the sequence. Can
        be either "forward" or "backward".
    recurrent_dropout_prob: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as:

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 direction: str = "forward",
                 recurrent_dropout_prob: float = 0.0,
                 use_highway: bool = True):
        super(AugmentedLstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.direction = direction
        self.use_highway = use_highway

        if use_highway:
            self.input_linearity = torch.nn.Linear(input_size, 6 * output_size, bias=True)
            self.state_linearity = torch.nn.Linear(output_size, 5 * output_size, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(input_size, 4 * output_size, bias=True)
            self.state_linearity = torch.nn.Linear(output_size, 4 * output_size, bias=True)

        self.dropout = Dropout(p=recurrent_dropout_prob, inplace=False) if recurrent_dropout_prob > 0.0 else None

    def reset_parameters(self):
        # TODO: Orthonormal initialization
        pass

    def forward(self, input: PackedSequence):
        assert isinstance(input, PackedSequence), 'input must be PackedSequence but got %s'%(type(input))

        input, batch_lengths = pad_packed_sequence(input, batch_first=True)

        batch_size = input.size()[0]
        total_timesteps = input.size()[1]

        output_accumulator = Variable(input.data.new().resize_(batch_size, total_timesteps, self.output_size).fill_(0))
        full_batch_previous_memory = Variable(input.data.new().resize_(batch_size, self.output_size).fill_(0))
        full_batch_previous_state = Variable(input.data.new().resize_(batch_size, self.output_size).fill_(0))

        current_length_index = batch_size - 1 if self.direction == "forward" else 0

        for timestep in range(total_timesteps):

            # The index depends on which end we start.
            index = timestep if self.direction == "forward" else total_timesteps - timestep - 1

            # This next bit of code is very clever, truly inhumane and is
            # actually the way Pytorch itself implements RNNs on GPUs. What
            # we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that:
            #
            #    ****   the batch inputs must be _ordered_ by length   ****
            #    **** from longest (first in batch) to shortest (last) ****
            #
            # so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.

            if self.direction == "forward":
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1

            # If we're going backwards, we are _picking up_ more indices.
            elif self.direction == "backward":
                # Complicated logic, hence the overly verbose while loop variables.
                not_at_max_number_of_sequences = current_length_index < (len(batch_lengths) - 1)
                next_shortest_sequence_uses_this_timestep = batch_lengths[current_length_index + 1] > index

                while not_at_max_number_of_sequences and next_shortest_sequence_uses_this_timestep:
                    current_length_index += 1
                    not_at_max_number_of_sequences = current_length_index < (len(batch_lengths) - 1)
                    next_shortest_sequence_uses_this_timestep = batch_lengths[current_length_index + 1] > index

            # Actually get the slices of the batch which we need for
            # the computation at this timestep.
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = input[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once to
            # make use of GPU parallelism.
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.input_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and input.
            input_gate = torch.sigmoid(projected_input[:, 0 * self.output_size: 1 * self.output_size] +
                                       projected_state[:, 0 * self.output_size: 1 * self.output_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.output_size: 2 * self.output_size] +
                                        projected_state[:, 1 * self.output_size: 2 * self.output_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.output_size: 3 * self.output_size] +
                                     projected_state[:, 2 * self.output_size: 3 * self.output_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.output_size: 4 * self.output_size] +
                                        projected_state[:, 3 * self.output_size: 4 * self.output_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * torch.tanh(memory)

            if self.use_highway:
                highway_gate = torch.sigmoid(projected_input[:, 4 * self.output_size: 5 * self.output_size] +
                                             projected_state[:, 4 * self.output_size: 5 * self.output_size])
                highway_input_projection = projected_input[:, 5 * self.output_size: 6 * self.output_size]
                timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

            if self.dropout:
                timestep_output = self.dropout(timestep_output)

            full_batch_previous_memory = Variable(input.data.new().resize_(batch_size, self.output_size).fill_(0))
            full_batch_previous_state = Variable(input.data.new().resize_(batch_size, self.output_size).fill_(0))
            full_batch_previous_memory[0: current_length_index + 1] = memory
            full_batch_previous_state[0: current_length_index + 1] = timestep_output
            output_accumulator[0: current_length_index + 1, index] = timestep_output

        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)
        return output_accumulator
