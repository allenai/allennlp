
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.autograd import Function, Variable
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode('utf-8'))
SRU_PTX = SRU_PROG.compile()
SRU_MOD = function.Module()
SRU_MOD.load(bytes(SRU_PTX.encode()))
SRU_FWD_FUNC = SRU_MOD.get_function('sru_fwd')
SRU_BWD_FUNC = SRU_MOD.get_function('sru_bwd')
SRU_BiFWD_FUNC = SRU_MOD.get_function('sru_bi_fwd')
SRU_BiBWD_FUNC = SRU_MOD.get_function('sru_bi_bwd')

Stream = namedtuple('Stream', ['ptr'])
SRU_STREAM = Stream(ptr=torch.cuda.current_stream().cuda_stream)


class _SruFunction(Function):

    def __init__(self, use_tanh, hidden_size, bidirectional=False):
        super(_SruFunction, self).__init__()
        self.use_tanh = use_tanh
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, u, x, bias, init=None, mask_h=None):
        output_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // self.hidden_size
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * output_size
        threads_per_block = min(512, ncols)
        num_blocks = (ncols - 1) / threads_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, output_size) if x.dim() == 3 else (batch, output_size)
        c = x.new(*size)
        h = x.new(*size)

        kernel = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        kernel_args = [u.contiguous().data_ptr(),
                       x.contiguous().data_ptr() if k_ == 3 else 0,
                       bias.data_ptr(),
                       init_.contiguous().data_ptr(),
                       mask_h.data_ptr() if mask_h is not None else 0,
                       length,
                       batch,
                       self.hidden_size,
                       k_,
                       h.data_ptr(),
                       c.data_ptr(),
                       self.use_tanh]
        kernel(kernel_args,
               block=(threads_per_block, 1, 1),
               grid=(num_blocks, 1, 1),
               stream=SRU_STREAM)

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1,:,:d], c[0,:,d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        output_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // self.hidden_size
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * output_size
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) / thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, output_size)
        grad_init = x.new(batch, output_size)

        grad_x = x.new(*x.size()) if k_ == 3 else None
        backward_kernel = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC

        kernel_args = [u.contiguous().data_ptr(),
                       x.contiguous().data_ptr() if k_ == 3 else 0,
                       bias.data_ptr(),
                       init_.contiguous().data_ptr(),
                       mask_h.data_ptr() if mask_h is not None else 0,
                       c.data_ptr(),
                       grad_h.contiguous().data_ptr(),
                       grad_last.contiguous().data_ptr(),
                       length,
                       batch,
                       self.hidden_size,
                       k_,
                       grad_u.data_ptr(),
                       grad_x.data_ptr() if k_ == 3 else 0,
                       grad_bias.data_ptr(),
                       grad_init.data_ptr(),
                       self.use_tanh]

        backward_kernel(kernel_args,
                        block=(thread_per_block, 1, 1),
                        grid=(num_block, 1, 1),
                        stream=SRU_STREAM)
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0,
                 recurrent_dropout_probability: float = 0,
                 use_tanh=1,
                 bidirectional: bool = False):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_tanh = use_tanh

        output_size = hidden_size * 2 if bidirectional else hidden_size
        k = 4 if input_size != output_size else 3
        size_per_dir = hidden_size * k
        self.weight = nn.Parameter(torch.Tensor(input_size, size_per_dir * 2 if bidirectional else size_per_dir))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4 if bidirectional else hidden_size * 2))
        self.reset_parameters()

    def reset_parameters(self):
        val_range = (3.0 / self.input_size) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

        if self.bidirectional:
            self.bias.data[self.hidden_size * 2:].zero_().fill_(1.0)
        else:
            self.bias.data[self.hidden_size:].zero_().fill_(1.0)

    def forward(self, inputs, initial_state=None):
        assert inputs.dim() == 2 or inputs.dim() == 3
        batch = inputs.size(-2)
        state_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2
        if initial_state is None:
            initial_state = Variable(inputs.data.new(batch, state_size).zero_())

        if self.training and (self.recurrent_dropout_probability > 0):
            dropout_mask = self.get_dropout_mask_((batch, self.input_size), self.recurrent_dropout_probability)
            inputs = inputs * dropout_mask.expand_as(inputs)

        two_dim_inputs = inputs if inputs.dim() == 2 else inputs.contiguous().view(-1, self.input_size)
        projected_inputs = two_dim_inputs.mm(self.weight)

        if self.training and (self.dropout > 0):
            mask_h = self.get_dropout_mask_((batch, state_size), self.dropout)
            cell_function = _SruFunction(self.use_tanh, self.hidden_size, self.bidirectional)
            outputs, state = cell_function(projected_inputs, inputs, self.bias, initial_state, mask_h)
        else:
            cell_function = _SruFunction(self.use_tanh, self.hidden_size, self.bidirectional)
            outputs, state = cell_function(projected_inputs, inputs, self.bias, initial_state)

        return outputs, state

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))


class SRU(nn.Module):
    """
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the SRU.
    hidden_size : int, required
        The dimension of the outputs of the SRU.
    num_layers : int, required
        The number of stacked SRUs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 recurrent_dropout_probability: float = 0.0,
                 use_tanh=1,
                 bidirectional: bool = False):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.sru_layers = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size*2 if bidirectional else hidden_size

        for i in range(num_layers):
            layer = SRUCell(input_size=self.input_size if i == 0 else self.out_size,
                            hidden_size=self.hidden_size,
                            dropout=dropout if i+1 != num_layers else 0,
                            recurrent_dropout_probability=self.recurrent_dropout_probability,
                            use_tanh=use_tanh,
                            bidirectional=bidirectional)
            self.sru_layers.append(layer)

    def forward(self,
                inputs: PackedSequence,
                initial_state: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : torch.Tensor, optional, (default = None)
            A tensor representing the initial hidden state and memory
            of the SRU, with shape (num_layers, batch_size, output_dimension).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        inputs, lengths = pad_packed_sequence(inputs, batch_first=True)

        # Kernel uses sequence_length as the first dimension, so we transpose here.
        inputs.transpose(0, 1)
        direction = 2 if self.bidirectional else 1
        if initial_state is None:
            zeros = Variable(inputs.data.new(inputs.size(1), self.hidden_size * direction).zero_())
            initial_state = [zeros for _ in range(self.num_layers)]
        else:
            initial_state = initial_state.chunk(self.num_layers, 0)

        layer_output = inputs
        final_states = []
        for i, layer in enumerate(self.sru_layers):
            output, state = layer(layer_output, initial_state[i])
            layer_output = output
            final_states.append(state)

        # Returned output from kernel is sequence dimension first,
        # so we transpose back to a batch first tensor.
        layer_output = pack_padded_sequence(layer_output.transpose(0, 1), lengths)
        return layer_output, torch.stack(final_states)
