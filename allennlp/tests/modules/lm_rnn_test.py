# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch

from allennlp.modules import LMRNN
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data import Vocabulary


class TestLMRNN(AllenNlpTestCase):

    def test_from_params_builders_LMRNN_correctly(self):

        params = Params({
                "type": "vanilla_RNN",
                "layer_num": 3,
                "unit": 'gru',
                "input_dim": 3,
                "hid_dim": 5,
                "dropout": 0.5,
                "batch_norm": False
                })

        rnn = LMRNN.from_params(params)
        assert rnn.__class__.__name__ == 'vanilla_RNN'
        assert rnn.input_dim == 3
        assert rnn.output_dim == 5
        for ind, tup in enumerate(rnn.layer.children()):
            if 0==ind:
                assert tup.input_dim == 3
            else:
                assert tup.input_dim == 5
            assert tup.output_dim == 5
            assert tup.batch_norm == False
            assert tup.dropout == 0.5
            assert type(tup.layer) == torch.nn.GRU
            assert tup.hidden_state is None
