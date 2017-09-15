import torch
import numpy
from torch.autograd import Function, NestedIOFunction, Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from ._ext import highway_lstm_layer

class HighwayLSTMFunction(NestedIOFunction):
    def __init__(self, input_size, hidden_size, num_layers=1,
            recurrent_dropout_prob=0, train=True):
        super(HighwayLSTMFunction, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.train = train

    def forward_extended(self, input, weight, bias, hy, cy, dropout, lengths, gates):
        self.seq_length, self.mini_batch, self.input_size = input.size()

        tmp_i = input.new(self.mini_batch, 6 * self.hidden_size)
        tmp_h = input.new(self.mini_batch, 5 * self.hidden_size)


        highway_lstm_layer.highway_lstm_forward_cuda(
                self.input_size, self.hidden_size, self.mini_batch, self.num_layers,
                self.seq_length, input, lengths, hy, cy, tmp_i, tmp_h, weight,
                bias, dropout, self.recurrent_dropout_prob, gates, 1 if self.train else 0)

        self.save_for_backward(input, lengths, weight, bias, hy, cy, dropout, gates)

        output = hy[-1,1:]

        return output, hy[:,1:]

    def backward(self, grad_output, grad_hy):
        input, lengths, weight, bias, hy, cy, dropout, gates = self.saved_tensors
        input = input.contiguous()
        grad_input, grad_weight, grad_bias, grad_hx, grad_cx, grad_dropout, grad_lengths, grad_gates = None, None, None, None, None, None, None, None

        grad_input = input.new().resize_as_(input).zero_()
        grad_hx = input.new().resize_as_(hy).zero_()
        grad_cx = input.new().resize_as_(cy).zero_()

        grad_weight = input.new()
        grad_bias = input.new()
        ones = input.new()
        if self.needs_input_grad[1]:
            grad_weight.resize_as_(weight).zero_()
            grad_bias.resize_as_(bias).zero_()
            ones.resize_(self.mini_batch).fill_(1)

        tmp_i_gates_grad = input.new().resize_(self.mini_batch, 6 * self.hidden_size).zero_()
        tmp_h_gates_grad = input.new().resize_(self.mini_batch, 5 * self.hidden_size).zero_()

        highway_lstm_layer.highway_lstm_backward_cuda(
                self.input_size, self.hidden_size, self.mini_batch, self.num_layers,
                self.seq_length, grad_output, lengths, grad_hx, grad_cx, input, 
                hy, cy, weight, gates, dropout, self.recurrent_dropout_prob, 
                tmp_h_gates_grad, tmp_i_gates_grad, ones, grad_hy, grad_input,
                grad_weight, grad_bias, 1 if self.train else 0, 1 if self.needs_input_grad[1] else 0)

        return grad_input, grad_weight, grad_bias, grad_hx, grad_cx, grad_dropout, grad_lengths, grad_gates

class HighwayLSTMLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, recurrent_dropout_prob=0):
        super(HighwayLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.batch_first = batch_first
        self.training = True

        self.ih_size = 6 * hidden_size
        self.hh_size = 5 * hidden_size
        self.bias_size = 5 * hidden_size

        assert num_layers > 0

        weight_size = 0
        bias_size = 0
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            ih_weights = self.ih_size * layer_input_size
            hh_weights = self.hh_size * hidden_size
            weight_size += ih_weights + hh_weights
            
            if bias:
                bias_size += self.bias_size

        self.weight = Parameter(torch.FloatTensor(weight_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(bias_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        weight_index = 0
        bias_index = 0
        for i in range(self.num_layers):
            insize = self.input_size if i == 0 else self.hidden_size
            i_weights = block_orthonormal_initialization(insize, self.hidden_size, 6)
            self.weight.data[weight_index:weight_index + i_weights.nelement()].copy_(i_weights)
            weight_index += i_weights.nelement()

            h_weights = block_orthonormal_initialization(self.hidden_size, self.hidden_size, 5)
            self.weight.data[weight_index:weight_index + h_weights.nelement()].copy_(h_weights)
            weight_index += h_weights.nelement()

            # forget bias
            self.bias.data[bias_index+self.hidden_size:bias_index+2*self.hidden_size].fill_(1)
            bias_index += 5 * self.hidden_size

    def forward(self, input, dropout_weights=None):

        assert isinstance(input, PackedSequence), "HighwayLSTMLayer only accepts PackedSequence input"
        input, lengths = pad_packed_sequence(input, batch_first = self.batch_first)

        if self.batch_first:
            input = input.transpose(0,1)

        self.seq_length, self.mini_batch, input_size = input.size()
        assert input_size == self.input_size

        hy = Variable(input.data.new(self.num_layers, (self.seq_length + 1), self.mini_batch, self.hidden_size).zero_(), requires_grad=False)
        cy = Variable(input.data.new(self.num_layers, (self.seq_length + 1), self.mini_batch, self.hidden_size).zero_(), requires_grad=False)

        if dropout_weights is None:
            dropout_weights = input.data.new()
            if self.training:
                dropout_weights.resize_(self.num_layers, self.mini_batch, self.hidden_size)
                dropout_weights.bernoulli_(1 - self.recurrent_dropout_prob)
            dropout_weights = Variable(dropout_weights, requires_grad = False)

        gates = Variable(input.data.new().resize_(self.num_layers, self.seq_length, self.mini_batch, 6 * self.hidden_size))

        lengths_var = Variable(torch.IntTensor(lengths))

        output, hidden = HighwayLSTMFunction(self.input_size, self.hidden_size, num_layers=self.num_layers, recurrent_dropout_prob=self.recurrent_dropout_prob, train=self.training)(input, self.weight, self.bias, hy, cy, dropout_weights, lengths_var, gates)

        output = output.transpose(0,1)

        output = pack_padded_sequence(output, lengths, batch_first = self.batch_first)

        return output, hidden


if __name__ == '__main__':
    # Test
    lstm = HighwayLSTMLayer(3, 6, num_layers=1).cuda()

    input = Variable(torch.randn(5,7,3).cuda())

    output, hidden = lstm(input)

def orthonormal_initialization(dim_in, dim_out, factor=1.0, seed=None, dtype='float64'):
    rng = numpy.random.RandomState(seed)
    if dim_in == dim_out:
        M = rng.randn(*[dim_in, dim_out]).astype(dtype)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        param = torch.Tensor(Q * factor)
    else:
        M1 = rng.randn(dim_in, dim_in).astype(dtype)
        M2 = rng.randn(dim_out, dim_out).astype(dtype)
        Q1, R1 = numpy.linalg.qr(M1)
        Q2, R2 = numpy.linalg.qr(M2)
        Q1 = Q1 * numpy.sign(numpy.diag(R1))
        Q2 = Q2 * numpy.sign(numpy.diag(R2))
        n_min = min(dim_in, dim_out)
        param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * factor
        param = torch.Tensor(param)
    return param

def block_orthonormal_initialization(dim_in, dim_out, num_blocks, factor=1.0, seed=None, dtype='float64'):
    param = torch.cat([orthonormal_initialization(dim_in, dim_out) for i in range(num_blocks)], 1)
    return param

