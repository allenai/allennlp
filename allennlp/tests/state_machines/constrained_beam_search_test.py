# pylint: disable=invalid-name,no-self-use,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines import ConstrainedBeamSearch
from .simple_transition_system import SimpleState, SimpleTransitionFunction


class TestConstrainedBeamSearch(AllenNlpTestCase):
    def test_search(self):
        # The simple transition system starts at some number, adds one or two at each state, and
        # tries to get to 4.  The highest scoring path has the shortest length and the highest
        # numbers (so always add two, unless you're at 3).  From -3, there are lots of possible
        # sequences: [-2, -1, 0, 1, 2, 3, 4], [-1, 1, 3, 4], ...  We'll specify a few of those up
        # front as "allowed", and use that to test the constrained beam search implementation.
        initial_state = SimpleState([0], [[]], [torch.Tensor([0.0])], [-3])
        beam_size = 3
        # pylint: disable=bad-whitespace
        allowed_sequences = torch.Tensor([[[-2, -1, 0, 1,  2,  3,  4],
                                           [-2,  0, 2, 4, -1, -1, -1],
                                           [-1,  1, 3, 4, -1, -1, -1],
                                           [-2, -1, 0, 1,  2,  4, -1],
                                           [-1,  0, 1, 2,  3,  4, -1],
                                           [-1,  1, 2, 3,  4, -1, -1]]])
        # pylint: enable=bad-whitespace
        mask = torch.Tensor([[[1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1, 0, 0]]])

        beam_search = ConstrainedBeamSearch(beam_size, allowed_sequences, mask)

        # Including the value in the score will make us pick states that have higher numbers first.
        # So with a beam size of 3, we'll get all of the states that start with `-1` after the
        # first step, even though in the end one of the states that starts with `-2` is better than
        # two of the states that start with `-1`.
        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(initial_state, decoder_step)

        assert len(best_states) == 1
        assert best_states[0][0].action_history[0] == [-1, 1, 3, 4]
        assert best_states[0][1].action_history[0] == [-1, 1, 2, 3, 4]
        assert best_states[0][2].action_history[0] == [-1, 0, 1, 2, 3, 4]

        # With a beam size of 6, we should get the other allowed path of length 4 as the second
        # best result.
        beam_size = 6
        beam_search = ConstrainedBeamSearch(beam_size, allowed_sequences, mask)
        decoder_step = SimpleTransitionFunction(include_value_in_score=True)
        best_states = beam_search.search(initial_state, decoder_step)

        assert len(best_states) == 1
        assert best_states[0][0].action_history[0] == [-1, 1, 3, 4]
        assert best_states[0][1].action_history[0] == [-2, 0, 2, 4]
