import torch
from torch.autograd import NestedIOFunction, Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from allennlp.nn.initializers import block_orthogonal
from allennlp.custom_extensions._ext import highway_lstm_layer


class _HighwayLSTMFunction(NestedIOFunction):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 train: bool = True):
        super(_HighwayLSTMFunction, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.train = train

    def forward_extended(self, input, weight, bias, hy, cy, dropout, lengths, gates):
        sequence_length, batch_size, input_size = input.size()
        tmp_i = input.new(batch_size, 6 * self.hidden_size)
        tmp_h = input.new(batch_size, 5 * self.hidden_size)

        highway_lstm_layer.highway_lstm_forward_cuda(input_size,
                                                     self.hidden_size,
                                                     batch_size,
                                                     self.num_layers,
                                                     sequence_length,
                                                     input,
                                                     lengths,
                                                     hy,
                                                     cy,
                                                     tmp_i,
                                                     tmp_h,
                                                     weight,
                                                     bias,
                                                     dropout,
                                                     gates,
                                                     1 if self.train else 0)

        self.save_for_backward(input, lengths, weight, bias, hy, cy, dropout, gates)

        output = hy[-1, 1:]
        return output, hy[:, 1:]

    def backward(self, grad_output, grad_hy):
        input, lengths, weight, bias, hy, cy, dropout, gates = self.saved_tensors
        input = input.contiguous()
        sequence_length, batch_size, input_size = input.size()

        grad_input = input.new().resize_as_(input).zero_()
        grad_hx = input.new().resize_as_(hy).zero_()
        grad_cx = input.new().resize_as_(cy).zero_()
        grad_weight = input.new()
        grad_bias = input.new()
        grad_dropout = None
        grad_lengths = None
        grad_gates = None

        if self.needs_input_grad[1]:
            grad_weight.resize_as_(weight).zero_()
            grad_bias.resize_as_(bias).zero_()

        tmp_i_gates_grad = input.new().resize_(batch_size, 6 * self.hidden_size).zero_()
        tmp_h_gates_grad = input.new().resize_(batch_size, 5 * self.hidden_size).zero_()

        is_training = 1 if self.train else 0
        highway_lstm_layer.highway_lstm_backward_cuda(
            input_size, self.hidden_size, batch_size,
            self.num_layers, sequence_length, grad_output,
            lengths, grad_hx, grad_cx, input, hy, cy, weight,
            gates, dropout, tmp_h_gates_grad, tmp_i_gates_grad,
            grad_hy, grad_input, grad_weight, grad_bias, is_training, 1 if self.needs_input_grad[1] else 0)

        return grad_input, grad_weight, grad_bias, grad_hx, grad_cx, grad_dropout, grad_lengths, grad_gates


class HighwayLSTM(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_prob: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .

    Returns
    -------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 recurrent_dropout_prob: float = 0):
        super(HighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.training = True

        self.ih_size = 6 * hidden_size
        self.hh_size = 5 * hidden_size
        self.bias_size = 5 * hidden_size

        weight_size = 0
        bias_size = 0
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            ih_weights = self.ih_size * layer_input_size
            hh_weights = self.hh_size * hidden_size
            weight_size += ih_weights + hh_weights

            bias_size += self.bias_size

        self.weight = Parameter(torch.FloatTensor(weight_size))
        self.bias = Parameter(torch.FloatTensor(bias_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        weight_index = 0
        bias_index = 0
        for i in range(self.num_layers):
            input_size = self.input_size if i == 0 else self.hidden_size

            # Create a tensor of the right size and initialize it.
            init_tensor = self.weight.data.new(input_size, self.hidden_size * 6).zero_()
            block_orthogonal(init_tensor, [self.input_size, self.hidden_size])
            # Copy it into the flat weight.
            self.weight.data[weight_index: weight_index + init_tensor.nelement()].view_as(init_tensor).copy_(init_tensor)
            weight_index += init_tensor.nelement()

            # Same for the recurrent connection weight.
            init_tensor = self.weight.data.new(input_size, self.hidden_size * 5).zero_()
            block_orthogonal(init_tensor, [self.input_size, self.hidden_size])
            self.weight.data[weight_index: weight_index + init_tensor.nelement()].view_as(init_tensor).copy_(init_tensor)
            weight_index += init_tensor.nelement()

            # forget bias
            self.bias.data[bias_index + self.hidden_size:bias_index + 2 * self.hidden_size].fill_(1)
            bias_index += 5 * self.hidden_size

    def forward(self, inputs: PackedSequence,
                initial_state: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            Currently, this is ignored.

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        inputs, lengths = pad_packed_sequence(inputs, batch_first=True)

        # Kernel takes sequence length first tensors.
        inputs = inputs.transpose(0, 1)

        sequence_length, batch_size, input_size = inputs.size()
        accumulator_shape = [self.num_layers, sequence_length + 1, batch_size, self.hidden_size]
        state_accumulator = Variable(inputs.data.new(*accumulator_shape).zero_(), requires_grad=False)
        memory_accumulator = Variable(inputs.data.new(*accumulator_shape).zero_(), requires_grad=False)

        dropout_weights = inputs.data.new().resize_(self.num_layers, batch_size, self.hidden_size).fill_(1.0)
        if self.training:
            dropout_weights.bernoulli_(1 - self.recurrent_dropout_prob).div_((1 - self.recurrent_dropout_prob))

        dropout_weights = Variable(dropout_weights, requires_grad=False)

        gates = Variable(inputs.data.new().resize_(self.num_layers,
                                                   sequence_length,
                                                   batch_size,
                                                   6 * self.hidden_size))

        lengths_var = Variable(torch.IntTensor(lengths))
        implementation = _HighwayLSTMFunction(self.input_size,
                                              self.hidden_size,
                                              num_layers=self.num_layers,
                                              train=self.training)
        output, hidden = implementation(inputs, self.weight, self.bias, state_accumulator,
                                        memory_accumulator, dropout_weights, lengths_var, gates)

        output = output.transpose(0, 1)
        output = pack_padded_sequence(output, lengths, batch_first=True)
        return output, hidden

