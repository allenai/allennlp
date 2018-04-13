# pylint: disable=invalid-name,no-self-use,protected-access
import math

from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import DecoderState
from allennlp.nn.decoding.decoder_trainers import MaximumMarginalLikelihood
from ..simple_transition_system import SimpleDecoderState, SimpleDecoderStep


class TestMaximumMarginalLikelihood(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.initial_state = SimpleDecoderState([0, 1],
                                                [[], []],
                                                [Variable(torch.Tensor([0.0])), Variable(torch.Tensor([0.0]))],
                                                [0, 1])
        self.decoder_step = SimpleDecoderStep()
        self.targets = torch.autograd.Variable(torch.Tensor([[[2, 3, 4], [1, 3, 4], [1, 2, 4]],
                                                             [[3, 4, 0], [2, 3, 4], [0, 0, 0]]]))
        self.target_mask = torch.autograd.Variable(torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                                                 [[1, 1, 0], [1, 1, 1], [0, 0, 0]]]))

        self.supervision = (self.targets, self.target_mask)
        # High beam size ensures exhaustive search.
        self.trainer = MaximumMarginalLikelihood()

    def test_decode(self):
        decoded_info = self.trainer.decode(self.initial_state, self.decoder_step, self.supervision)

        # Our loss is the negative log sum of the scores from each target sequence.  The score for
        # each sequence in our simple transition system is just `-sequence_length`.
        instance0_loss = math.log(math.exp(-3) * 3)  # all three sequences have length 3
        instance1_loss = math.log(math.exp(-2) + math.exp(-3))  # one has length 2, one has length 3
        expected_loss = -(instance0_loss + instance1_loss) / 2
        assert_almost_equal(decoded_info['loss'].data.numpy(), expected_loss)

    def test_create_allowed_transitions(self):
        result = self.trainer._create_allowed_transitions(self.targets, self.target_mask)
        # There were two instances in this batch.
        assert len(result) == 2

        # The first instance had six valid action sequence prefixes.
        assert len(result[0]) == 6
        assert result[0][()] == {1, 2}
        assert result[0][(1,)] == {2, 3}
        assert result[0][(1, 2)] == {4}
        assert result[0][(1, 3)] == {4}
        assert result[0][(2,)] == {3}
        assert result[0][(2, 3)] == {4}

        # The second instance had four valid action sequence prefixes.
        assert len(result[1]) == 4
        assert result[1][()] == {2, 3}
        assert result[1][(2,)] == {3}
        assert result[1][(2, 3)] == {4}
        assert result[1][(3,)] == {4}

    def test_get_allowed_actions(self):
        state = DecoderState([0, 1, 0], [[1], [0], []], [])
        allowed_transitions = [{(1,): {2}, (): {3}}, {(0,): {4, 5}}]
        allowed_actions = self.trainer._get_allowed_actions(state, allowed_transitions)
        assert allowed_actions == [{2}, {4, 5}, {3}]
