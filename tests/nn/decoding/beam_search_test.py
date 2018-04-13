# pylint: disable=invalid-name,no-self-use,protected-access
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import BeamSearch
from .simple_transition_system import SimpleDecoderState, SimpleDecoderStep


class TestBeamSearch(AllenNlpTestCase):
    def test_search(self):
        beam_search = BeamSearch.from_params(Params({'beam_size': 4}))
        initial_state = SimpleDecoderState([0, 1, 2, 3],
                                           [[], []],
                                           [Variable(torch.Tensor([0.0])), Variable(torch.Tensor([0.0]))],
                                           [-3, 1, -20, 5])
        decoder_step = SimpleDecoderStep(include_value_in_score=True)
        best_states = beam_search.search(5,
                                         initial_state,
                                         decoder_step,
                                         keep_final_unfinished_states=False)

        # Instance with batch index 2 needed too many steps to finish, and batch index 3 had no
        # path to get to a finished state.  (See the simple transition system definitely; goal is
        # to end up at 4, actions are either add one or two to starting value.)
        assert len(best_states) == 2
        print(list((x.action_history[0], x.score) for x in best_states[0]))
        assert best_states[0][0].action_history[0] == [-1, 1, 3, 4]
        assert best_states[1][0].action_history[0] == [3, 4]
