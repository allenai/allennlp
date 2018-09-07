# pylint: disable=invalid-name,no-self-use,protected-access
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines import BeamSearch
from .simple_transition_system import SimpleState, SimpleTransitionFunction


class TestBeamSearch(AllenNlpTestCase):
    def test_search(self):
        beam_search = BeamSearch.from_params(Params({'beam_size': 4}))
        initial_state = SimpleState([0, 1, 2, 3],
                                    [[], [], [], []],
                                    [torch.Tensor([0.0]),
                                     torch.Tensor([0.0]),
                                     torch.Tensor([0.0]),
                                     torch.Tensor([0.0])],
                                    [-3, 1, -20, 5])
        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(5,
                                         initial_state,
                                         decoder_step,
                                         keep_final_unfinished_states=False)

        # Instance with batch index 2 needed too many steps to finish, and batch index 3 had no
        # path to get to a finished state.  (See the simple transition system definition; goal is
        # to end up at 4, actions are either add one or two to starting value.)
        assert len(best_states) == 2
        assert best_states[0][0].action_history[0] == [-1, 1, 3, 4]
        assert best_states[1][0].action_history[0] == [3, 4]

        best_states = beam_search.search(5,
                                         initial_state,
                                         decoder_step,
                                         keep_final_unfinished_states=True)

        # Now we're keeping final unfinished states, which allows a "best state" for the instances
        # that didn't have one before.  Our previous best states for the instances that finish
        # doesn't change, because the score for taking another step is always negative at these
        # values.
        assert len(best_states) == 4
        assert best_states[0][0].action_history[0] == [-1, 1, 3, 4]
        assert best_states[1][0].action_history[0] == [3, 4]
        assert best_states[2][0].action_history[0] == [-18, -16, -14, -12, -10]
        assert best_states[3][0].action_history[0] == [7, 9, 11, 13, 15]
