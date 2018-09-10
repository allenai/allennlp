# pylint: disable=invalid-name,no-self-use,protected-access
import math

from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from ..simple_transition_system import SimpleState, SimpleTransitionFunction


class TestMaximumMarginalLikelihood(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.initial_state = SimpleState([0, 1],
                                         [[], []],
                                         [torch.Tensor([0.0]), torch.Tensor([0.0])],
                                         [0, 1])
        self.decoder_step = SimpleTransitionFunction()
        self.targets = torch.Tensor([[[2, 3, 4], [1, 3, 4], [1, 2, 4]],
                                     [[3, 4, 0], [2, 3, 4], [0, 0, 0]]])
        self.target_mask = torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                                         [[1, 1, 0], [1, 1, 1], [0, 0, 0]]])

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
