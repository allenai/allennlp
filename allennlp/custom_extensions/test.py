import sys

from baseline.lstm import AlternatingLSTM
from baseline.measurements import *
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from highway_lstm_layer import *

SMALL = False

B = 5 if SMALL else 83
L = 2 if SMALL else 8
T = 7 if SMALL else 101
I = 3 if SMALL else 103
O = 11 if SMALL else 311
DROPOUT = False
TRAIN = True

dp = 0.
if DROPOUT:
    dp = 0.5

baseline = AlternatingLSTM(I, O, L, recurrent_dropout_prob=dp).cuda()

#for n, p in baseline.named_parameters():
#    print n, p.data.size()
#    p.data.fill_(.0001)

mine = HighwayLSTMLayer(I, O, num_layers=L, recurrent_dropout_prob=dp).cuda()

#for n, p in mine.named_parameters():
#    print n, p.size()
#    p.data.fill_(.0001)
curr_weight_ind = 0
curr_bias_ind = 0
print mine.weight.nelement()
for x in xrange(L):
    print x, curr_weight_ind
    print curr_bias_ind
    x_weight = getattr(baseline, 'layer_%d'%x).xlin.weight
    h_weight = getattr(baseline, 'layer_%d'%x).hlin.weight
    bias = getattr(baseline, 'layer_%d'%x).hlin.bias
    mine.weight.data[curr_weight_ind:curr_weight_ind+x_weight.nelement()].copy_(x_weight.data.t())
    curr_weight_ind += x_weight.nelement()
    mine.weight.data[curr_weight_ind:curr_weight_ind+h_weight.nelement()].copy_(h_weight.data.t())
    curr_weight_ind += h_weight.nelement()
    mine.bias.data[curr_bias_ind:curr_bias_ind+bias.nelement()].copy_(bias.data)
    curr_bias_ind += bias.nelement()

if TRAIN:
    baseline.train()
    mine.train()
else:
    baseline.eval()
    mine.eval()

input = torch.randn(B,T,I).cuda()
input2 = input.clone()

baseline_input = Variable(input, requires_grad=True)
mine_input = Variable(input2, requires_grad=True)
lengths = [T-(i/2) for i in xrange(B)]
lengths = lengths[:B]
baseline_input_packed = pack_padded_sequence(baseline_input, lengths, batch_first=True)
mine_input_packed = pack_padded_sequence(mine_input, lengths, batch_first=True)

if DROPOUT:
    dropout = Variable(torch.Tensor(L,B,O).cuda().bernoulli_(dp))
else:
    dropout = None

with Timer('Baseline'):
    baseline_output = baseline(baseline_input_packed, dropout_weights = dropout)
    baseline_output, _ = pad_packed_sequence(baseline_output, batch_first = True)
    print baseline_output

with Timer("Mine"):
    mine_output, _ = mine(mine_input_packed, dropout_weights = dropout)
    mine_output, _ = pad_packed_sequence(mine_output, batch_first = True)
    print mine_output

diff = torch.max(baseline_output.data - mine_output.data)
assert diff < 1e-4, "Output does not match: " + str(diff)

back_err = torch.randn(B,T,O).cuda()

baseline.zero_grad()
baseline_output.backward(back_err)

mine.zero_grad()
mine_grad = mine_output.backward(back_err)

input_grad_diff = torch.max(baseline_input.grad.data - mine_input.grad.data)
assert input_grad_diff < 1e-4, "Input grad does not match: " + str(input_grad_diff)

weight_ind = 0
bias_ind = 0
for x in xrange(L):
    print "TEST %d"%(x)
    x_grad = getattr(baseline, 'layer_%d'%x).xlin.weight.grad
    h_grad = getattr(baseline, 'layer_%d'%x).hlin.weight.grad
    bias = getattr(baseline, 'layer_%d'%x).hlin.bias.grad

    mine_x_grad = mine.weight.grad[weight_ind:weight_ind+x_grad.nelement()].view(x_grad.size(1), x_grad.size(0)).t()
    weight_ind += x_grad.nelement()

    mine_h_grad = mine.weight.grad[weight_ind:weight_ind+h_grad.nelement()].view(h_grad.size(1), h_grad.size(0)).t()
    weight_ind += h_grad.nelement()

    mine_bias = mine.bias.grad[bias_ind:bias_ind+bias.nelement()]
    bias_ind += bias.nelement()

    x_diff = torch.max(mine_x_grad.data - x_grad.data)
    assert x_diff < 1e-4, "Layer %d x_weight does not match: "%x + str(x_diff)

    h_diff = torch.max(mine_h_grad.data - h_grad.data)
    assert h_diff < 1e-4, "Layer %d h_weight does not match: "%x + str(h_diff)

    bias_diff = torch.max(mine_bias.data - bias.data)
    assert bias_diff < 1e-4, "Layer %d bias does not match: "%x + str(bias_diff)

print "PASSED!"
