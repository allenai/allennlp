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

    def test_constraints(self):
        # The simple transition system starts at some number, adds one or two at each state, and
        # tries to get to 4.  The highest scoring path has the shortest length and the highest
        # numbers (so always add two, unless you're at 3).  From -3, there are lots of possible
        # sequences: [-2, -1, 0, 1, 2, 3, 4], [-1, 1, 3, 4], ...  We'll specify a few of those up
        # front as "allowed", and use that to test the constrained beam search implementation.
        initial_state = SimpleState([0], [[]], [torch.Tensor([0.0])], [-3])
        beam_size = 3
        initial_sequence = torch.Tensor([-2, -1, 0, 1])
        beam_search = BeamSearch(beam_size, initial_sequence=initial_sequence)

        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(7, initial_state, decoder_step)

        assert len(best_states) == 1

        # After the constraint runs out, we generate [3], [2],
        # then we generate [3, 5], [3, 4], [2, 4], the latter two of which are finished,
        # then we generate [3, 5, 7], [3, 5, 6], and we're out of steps, so we keep the former
        assert best_states[0][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
        assert best_states[0][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
        assert best_states[0][2].action_history[0] == [-2, -1, 0, 1, 3, 5, 7]

        # Now set the beam size to 6, we generate [3], [2]
        # then [3, 5], [2, 3], [3, 4], [2, 4] (the latter two of which are finished)
        # then [3, 5, 6], [3, 5, 7], [2, 3, 5], [2, 3, 4] (the last is finished)
        beam_size = 6
        beam_search = BeamSearch(beam_size, initial_sequence=initial_sequence, keep_beam_details=True)
        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(7, initial_state, decoder_step, keep_final_unfinished_states=False)

        assert len(best_states) == 1
        assert len(best_states[0]) == 3
        assert best_states[0][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
        assert best_states[0][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
        assert best_states[0][2].action_history[0] == [-2, -1, 0, 1, 2, 3, 4]

        # Check that beams are correct
        best_action_sequence = best_states[0][0].action_history[0]

        beam_snapshots = beam_search.beam_snapshots
        assert len(beam_snapshots) == 1

        beam_snapshots0 = beam_snapshots.get(0)
        assert beam_snapshots0 is not None

        for i, beam in enumerate(beam_snapshots0):
            assert all(len(sequence) == i + 1 for _, sequence in beam)
            if i < len(best_action_sequence):
                assert any(sequence[-1] == best_action_sequence[i] for _, sequence in beam)
