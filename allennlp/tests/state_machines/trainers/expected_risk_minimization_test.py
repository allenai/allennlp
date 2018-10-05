# pylint: disable=no-self-use,protected-access
import torch
import numpy as np
from numpy.testing import assert_almost_equal

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines.trainers import ExpectedRiskMinimization
from ..simple_transition_system import SimpleState, SimpleTransitionFunction


class TestExpectedRiskMinimization(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.initial_state = SimpleState([0], [[0]], [torch.Tensor([0.0])])
        self.decoder_step = SimpleTransitionFunction()
        # Cost is the number of odd elements in the action history.
        self.supervision = lambda state: torch.Tensor([sum([x%2 != 0 for x in
                                                            state.action_history[0]])])
        # High beam size ensures exhaustive search.
        self.trainer = ExpectedRiskMinimization(beam_size=100,
                                                normalize_by_length=False,
                                                max_decoding_steps=10)

    def test_get_finished_states(self):
        finished_states = self.trainer._get_finished_states(self.initial_state, self.decoder_step)
        state_info = [(state.action_history[0], state.score[0].item()) for state in finished_states]
        # There will be exactly five finished states with the following paths. Each score is the
        # negative of one less than the number of elements in the action history.
        assert len(finished_states) == 5
        assert ([0, 2, 4], -2) in state_info
        assert ([0, 1, 2, 4], -3) in state_info
        assert ([0, 1, 3, 4], -3) in state_info
        assert ([0, 2, 3, 4], -3) in state_info
        assert ([0, 1, 2, 3, 4], -4) in state_info

    def test_decode(self):
        decoded_info = self.trainer.decode(self.initial_state, self.decoder_step, self.supervision)
        # The best state corresponds to the shortest path.
        best_state = decoded_info['best_final_states'][0][0]
        assert best_state.action_history[0] == [0, 2, 4]
        # The scores and costs corresponding to the finished states will be
        # [0, 2, 4] : -2, 0
        # [0, 1, 2, 4] : -3, 1
        # [0, 1, 3, 4] : -3, 2
        # [0, 2, 3, 4] : -3, 1
        # [0, 1, 2, 3, 4] : -4, 2

        # This is the normalization factor while re-normalizing probabilities on the beam
        partition = np.exp(-2) + np.exp(-3) + np.exp(-3) + np.exp(-3) + np.exp(-4)
        expected_loss = ((np.exp(-2) * 0) + (np.exp(-3) * 1) + (np.exp(-3) * 2) +
                         (np.exp(-3) * 1) + (np.exp(-4) * 2)) / partition
        assert_almost_equal(decoded_info['loss'].data.numpy(), expected_loss)
