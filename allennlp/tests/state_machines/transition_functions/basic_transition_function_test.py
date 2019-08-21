# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import Attention
from allennlp.semparse.type_declarations.type_declaration import is_nonterminal
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.state_machines.transition_functions import BasicTransitionFunction


class BasicTransitionFunctionTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.decoder_step = BasicTransitionFunction(encoder_output_dim=2,
                                                    action_embedding_dim=2,
                                                    input_attention=Attention.by_name('dot_product')(),
                                                    add_action_bias=False)

        batch_indices = [0, 1, 0]
        action_history = [[1], [3, 4], []]
        score = [torch.FloatTensor([x]) for x in [.1, 1.1, 2.2]]
        hidden_state = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        memory_cell = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        previous_action_embedding = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        attended_question = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        # This maps non-terminals to valid actions, where the valid actions are grouped by _type_.
        # We have "global" actions, which are from the global grammar, and "linked" actions, which
        # are instance-specific and are generated based on question attention.  Each action type
        # has a tuple which is (input representation, output representation, action ids).
        valid_actions = {
                'e': {
                        'global': (torch.FloatTensor([[0, 0], [-1, -1], [-2, -2]]),
                                   torch.FloatTensor([[-1, -1], [-2, -2], [-3, -3]]),
                                   [0, 1, 2]),
                        'linked': (torch.FloatTensor([[.1, .2, .3], [.4, .5, .6]]),
                                   torch.FloatTensor([[3, 3], [4, 4]]),
                                   [3, 4])
                },
                'd': {
                        'global': (torch.FloatTensor([[0, 0]]),
                                   torch.FloatTensor([[-1, -1]]),
                                   [0]),
                        'linked': (torch.FloatTensor([[-.1, -.2, -.3], [-.4, -.5, -.6], [-.7, -.8, -.9]]),
                                   torch.FloatTensor([[5, 5], [6, 6], [7, 7]]),
                                   [1, 2, 3])
                }
        }
        grammar_state = [GrammarStatelet([nonterminal], valid_actions, is_nonterminal)
                         for _, nonterminal in zip(batch_indices, ['e', 'd', 'e'])]
        self.encoder_outputs = torch.FloatTensor([[[1, 2], [3, 4], [5, 6]], [[10, 11], [12, 13], [14, 15]]])
        self.encoder_output_mask = torch.FloatTensor([[1, 1, 1], [1, 1, 0]])
        self.possible_actions = [[('e -> f', False, None),
                                  ('e -> g', True, None),
                                  ('e -> h', True, None),
                                  ('e -> i', True, None),
                                  ('e -> j', True, None)],
                                 [('d -> q', True, None),
                                  ('d -> g', True, None),
                                  ('d -> h', True, None),
                                  ('d -> i', True, None)]]

        rnn_state = []
        for i in range(len(batch_indices)):
            rnn_state.append(RnnStatelet(hidden_state[i],
                                         memory_cell[i],
                                         previous_action_embedding[i],
                                         attended_question[i],
                                         self.encoder_outputs,
                                         self.encoder_output_mask))
        self.state = GrammarBasedState(batch_indices=batch_indices,
                                       action_history=action_history,
                                       score=score,
                                       rnn_state=rnn_state,
                                       grammar_state=grammar_state,
                                       possible_actions=self.possible_actions)

    def test_take_step(self):
        new_states = self.decoder_step.take_step(self.state,
                                                 max_actions=1,
                                                 allowed_actions=[{2, 3}, {0}, {4}])

        assert len(new_states) == 2
        new_state = new_states[0]
        assert new_state.batch_indices == [0]

        # We're not going to try to guess which action was taken (or set model weights so that we
        # know which action will be taken); we'll just check that we got one of the actions we were
        # expecting.
        expected_possibilities = set([((4,), ('j',)), ((1, 2), ('h',)), ((1, 3), ('i',))])
        actual = (tuple(new_state.action_history[0]), tuple(new_state.grammar_state[0]._nonterminal_stack))
        assert actual in expected_possibilities

        # These should just be copied from the prior state, no matter which action we took.
        assert_almost_equal(new_state.rnn_state[0].encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.cpu().numpy(),
                            self.encoder_output_mask.cpu().numpy())
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 1, we should have selected action 0 from group index 1 - there was
        # only one allowed action.
        assert new_state.batch_indices == [1]
        # These two have values taken from what's defined in setUp() - the prior action history
        # ([3, 4]) and the nonterminals corresponding to the action we picked ('q').
        assert new_state.action_history == [[3, 4, 0]]
        assert new_state.grammar_state[0]._nonterminal_stack == ['q']
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.rnn_state[0].encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.cpu().numpy(),
                            self.encoder_output_mask.cpu().numpy())
        assert new_state.possible_actions == self.possible_actions
