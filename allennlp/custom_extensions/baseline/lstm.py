import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class AlternatingLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, recurrent_dropout_prob=0.0, highway = True):
        super(AlternatingLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.out_dim = output_size
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            if i == 0:
                i_size = input_size
            else:
                i_size = output_size
            if i % 2 == 0:
                direction = 1
            else:
                direction = -1
            #layer = FastLSTMLayer(i_size, output_size, direction, recurrent_dropout_prob = recurrent_dropout_prob, highway = highway)
            layer = LSTMLayer(i_size, output_size, direction, recurrent_dropout_prob = recurrent_dropout_prob, highway = highway)
            setattr(self, 'layer_%d'%i, layer)
            layers.append(layer)
        self.layers = layers

    def forward(self, input, dropout_weights = None):
        if dropout_weights is not None:
            assert dropout_weights.size(0) == self.num_layers, "dropout_weights wrong size: %s"%(str(dropout_weights.size()))
        curr = input
        for i, l in enumerate(self.layers):
            if dropout_weights is not None:
                d = dropout_weights[i]
            else:
                d = None
            curr = l(curr, dropout_weights=d)

        return curr

class LSTMLayer(nn.Module):
    def __init__(self, input_size, output_size, direction, recurrent_dropout_prob = 0., highway = True):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.direction = direction
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.highway = highway

        if highway:
            self.xlin = nn.Linear(input_size, 6*output_size, bias=False)
            self.hlin = nn.Linear(output_size, 5*output_size, bias=True)
        else:
            self.xlin = nn.Linear(input_size, 4*output_size, bias=False)
            self.hlin = nn.Linear(output_size, 4*output_size, bias=True)
            

    def reset_parameters(self):
        # TODO: Orthonormal initialization
        pass

    def forward(self, input, dropout_weights=None):
        assert isinstance(input, PackedSequence), 'input must be PackedSequence but got %s'%(type(input))

        input, lengths = pad_packed_sequence(input, batch_first=True)

        B = input.size()[0]
        T = input.size()[1]

        assert B==len(lengths), 'T not equal len(lengths) (%d vs %d)'%(T, len(lengths))

        output_hs = Variable(input.data.new().resize_(B, T, self.output_size).fill_(0))
        prev_c = Variable(input.data.new().resize_(B, self.output_size).fill_(0))
        prev_h = Variable(input.data.new().resize_(B, self.output_size).fill_(0))
        curr_len = 0
        curr_len_index = B-1 if self.direction == 1 else 0

        if self.recurrent_dropout_prob > 0 and dropout_weights is None:
            dropout_weights = Variable(input.data.new().resize_(B, self.output_size))
            torch.rand(dropout_weights.size(), out=dropout_weights.data)
            dropout_weights = dropout_weights > self.recurrent_dropout_prob
            dropout_weights = dropout_weights.float()

        for ii in range(T):
            ind = ii if self.direction == 1 else T-ii-1

            if self.direction == 1:
                while lengths[curr_len_index] <= ind:
                    curr_len_index -= 1
            elif self.direction == -1:
                while curr_len_index < len(lengths)-1 and lengths[curr_len_index+1] > ind:
                    curr_len_index += 1 

            c_prev = prev_c[0:curr_len_index+1].clone()
            h_prev = prev_h[0:curr_len_index+1].clone()
            x_ = input[0:curr_len_index+1,ind]

            x2g = self.xlin(x_) # B x T x 6*O
            h2g = self.hlin(h_prev)

            i = F.sigmoid(x2g[:,0*self.output_size:1*self.output_size] + h2g[:,0*self.output_size:1*self.output_size])
            f = F.sigmoid(x2g[:,1*self.output_size:2*self.output_size] + h2g[:,1*self.output_size:2*self.output_size])
            c_init = F.tanh(x2g[:,2*self.output_size:3*self.output_size] + h2g[:,2*self.output_size:3*self.output_size])
            o = F.sigmoid(x2g[:,3*self.output_size:4*self.output_size] + h2g[:,3*self.output_size:4*self.output_size])
            c = i*c_init + f*c_prev

            h_prime = o * F.tanh(c)
            if self.highway:
                r = F.sigmoid(x2g[:,4*self.output_size:5*self.output_size] + h2g[:,4*self.output_size:5*self.output_size])
                h_hat = r*h_prime + (1 - r) * x2g[:,5*self.output_size:6*self.output_size]
            else:
                h_hat = h_prime

            if self.training and dropout_weights is not None:
                drop_w = dropout_weights[0:curr_len_index+1]
                h = h_hat * drop_w
            else:
                h = h_hat * (1 - self.recurrent_dropout_prob)

            prev_c = Variable(input.data.new().resize_(B, self.output_size).fill_(0))
            prev_h = Variable(input.data.new().resize_(B, self.output_size).fill_(0))
            prev_c[0:curr_len_index+1] = c
            prev_h[0:curr_len_index+1] = h
            output_hs[0:curr_len_index+1,ind] = h

        output_hs = pack_padded_sequence(output_hs, lengths, batch_first = True)
        return output_hs


