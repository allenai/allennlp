# pylint: disable=invalid-name,no-self-use,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines.interactive_beam_search import InteractiveBeamSearch
from .simple_transition_system import SimpleState, SimpleTransitionFunction

class TestInteractiveBeamSearch(AllenNlpTestCase):
    def test_constraints(self):
        # The simple transition system starts at some number, adds one or two at each state, and
        # tries to get to 4.  The highest scoring path has the shortest length and the highest
        # numbers (so always add two, unless you're at 3).  From -3, there are lots of possible
        # sequences: [-2, -1, 0, 1, 2, 3, 4], [-1, 1, 3, 4], ...  We'll specify a few of those up
        # front as "allowed", and use that to test the constrained beam search implementation.
        initial_state = SimpleState([0], [[]], [torch.Tensor([0.0])], [-3])
        beam_size = 3
        initial_sequence = torch.Tensor([-2, -1, 0, 1])
        beam_search = InteractiveBeamSearch(beam_size, initial_sequence)

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
        beam_search = InteractiveBeamSearch(beam_size, initial_sequence)
        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(7, initial_state, decoder_step, keep_final_unfinished_states=False)

        assert len(best_states) == 1
        assert len(best_states[0]) == 3
        assert best_states[0][0].action_history[0] == [-2, -1, 0, 1, 3, 4]
        assert best_states[0][1].action_history[0] == [-2, -1, 0, 1, 2, 4]
        assert best_states[0][2].action_history[0] == [-2, -1, 0, 1, 2, 3, 4]

        # Check that choices are correct
        allowed_steps = [[choice for _, choice in step_choices]
                         for step_choices in beam_search.choices]

        assert allowed_steps == [
                [-1, -2],  # from -3, can go to -1 or -2
                [0, -1],   # forced to -2, can go to 0 or -1
                [1, 0],    # forced to -1, can go to 1 or 0
                [2, 1],    # forced to  0, can go to 2 or 1
                [3, 2],    # forced to  1, can go to 3 or 2
                [5, 4, 3],     # could go 2 -> {3, 4} or 3 -> {4, 5}
                [7, 6, 5, 4]   # could go 3 -> {4, 5} or 4 -> {5, 6} or 5 -> {6, 7}
        ]

        # Check that beams are correct
        best_action_sequence = best_states[0][0].action_history[0]

        for i, beam in enumerate(beam_search.beams):
            assert all(len(sequence) == i + 1 for _, sequence in beam)
            if i < len(best_action_sequence):
                assert any(sequence[-1] == best_action_sequence[i] for _, sequence in beam)
