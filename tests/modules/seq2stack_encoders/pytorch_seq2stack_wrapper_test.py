# pylint: disable=no-self-use,invalid-name
import torch
from torch.autograd import Variable
from torch.nn import LSTM
from torch.nn.utils.rnn import pad_packed_sequence


from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2stack_encoders import PytorchSeq2StackWrapper


class MockStackedLSTM(torch.nn.Module):
    def __init__(self):
        super(MockStackedLSTM, self).__init__()
        self.lstms = [LSTM(bidirectional=True, num_layers=1, input_size=3,
                           hidden_size=7, batch_first=True),
                      LSTM(bidirectional=True, num_layers=1, input_size=14,
                           hidden_size=7, batch_first=True)]

    def forward(self, inputs, hidden_state):  # pylint: disable=arguments-differ
        output1, _ = self.lstms[0](inputs, hidden_state)
        output2, _ = self.lstms[1](output1, hidden_state)
        output1 = pad_packed_sequence(output1, batch_first=True)[0]
        output2 = pad_packed_sequence(output2, batch_first=True)[0]
        return torch.cat([output1.unsqueeze(0), output2.unsqueeze(0)], dim=0), None


class TestPytorchSeq2StackWrapper(AllenNlpTestCase):
    def test_wrapper_stacked_rnns(self):
        encoder = PytorchSeq2StackWrapper(MockStackedLSTM(), stateful=False)
        tensor = Variable(torch.rand([5, 8, 3]))
        mask = Variable(torch.ones(5, 8))
        mask.data[0, 3:] = 0
        stacked_output = encoder(tensor, mask)
        assert list(stacked_output.size()) == [2, 5, 8, 14]
