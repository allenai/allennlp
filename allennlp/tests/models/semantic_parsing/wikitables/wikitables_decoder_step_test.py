# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_step import WikiTablesDecoderStep
from allennlp.modules import SimilarityFunction
from allennlp.nn.decoding import GrammarState, RnnState
from allennlp.semparse.type_declarations.type_declaration import is_nonterminal


class WikiTablesDecoderStepTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.decoder_step = WikiTablesDecoderStep(encoder_output_dim=2,
                                                  action_embedding_dim=2,
                                                  attention_function=None,
                                                  num_start_types=3)

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
        grammar_state = [GrammarState([nonterminal], {}, valid_actions, {}, is_nonterminal)
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
            rnn_state.append(RnnState(hidden_state[i],
                                      memory_cell[i],
                                      previous_action_embedding[i],
                                      attended_question[i],
                                      self.encoder_outputs,
                                      self.encoder_output_mask))
        self.state = WikiTablesDecoderState(batch_indices=batch_indices,
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

    def test_compute_new_states_with_no_action_constraints(self):
        # pylint: disable=protected-access
        # This test is basically identical to the previous one, but without specifying
        # `allowed_actions`.  This makes sure we get the right behavior at test time.
        log_probs = torch.FloatTensor([[.1, .9, -.1, .2],
                                       [.3, 1.1, .1, .8],
                                       [.1, .25, .3, .4]])
        considered_actions = [[0, 1, 2, 3], [0, -1, 3, -1], [0, 2, 4, -1]]
        max_actions = 1
        step_action_embeddings = torch.FloatTensor([[[1, 1], [9, 9], [2, 2], [3, 3]],
                                                    [[4, 4], [9, 9], [3, 3], [9, 9]],
                                                    [[1, 1], [2, 2], [5, 5], [9, 9]]])
        new_hidden_state = torch.FloatTensor([[i + 1, i + 1] for i in range(len(considered_actions))])
        new_memory_cell = torch.FloatTensor([[i + 1, i + 1] for i in range(len(considered_actions))])
        new_attended_question = torch.FloatTensor([[i + 1, i + 1] for i in range(len(considered_actions))])
        new_attention_weights = torch.FloatTensor([[i + 1, i + 1] for i in range(len(considered_actions))])
        new_states = WikiTablesDecoderStep._compute_new_states(self.state,
                                                               log_probs,
                                                               new_hidden_state,
                                                               new_memory_cell,
                                                               step_action_embeddings,
                                                               new_attended_question,
                                                               new_attention_weights,
                                                               considered_actions,
                                                               allowed_actions=None,
                                                               max_actions=max_actions)

        assert len(new_states) == 2
        new_state = new_states[0]
        # For batch instance 0, we should have selected action 1 from group index 0.
        assert new_state.batch_indices == [0]
        assert_almost_equal(new_state.score[0].detach().cpu().numpy().tolist(), [.9])
        # These two have values taken from what's defined in setUp() - the prior action history
        # ([1]) and the nonterminals corresponding to the action we picked ('j').
        assert new_state.action_history == [[1, 1]]
        assert new_state.grammar_state[0]._nonterminal_stack == ['g']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.rnn_state[0].hidden_state.cpu().numpy().tolist(), [1, 1])
        assert_almost_equal(new_state.rnn_state[0].memory_cell.cpu().numpy().tolist(), [1, 1])
        assert_almost_equal(new_state.rnn_state[0].previous_action_embedding.cpu().numpy().tolist(), [9, 9])
        assert_almost_equal(new_state.rnn_state[0].attended_input.cpu().numpy().tolist(), [1, 1])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.rnn_state[0].encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.detach().cpu().numpy(),
                            self.encoder_output_mask.detach().cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 0, we should have selected action 0 from group index 1.
        assert new_state.batch_indices == [1]
        assert_almost_equal(new_state.score[0].detach().cpu().numpy().tolist(), [.3])
        # These have values taken from what's defined in setUp() - the prior action history
        # ([3, 4]) and the nonterminals corresponding to the action we picked ('q').
        assert new_state.action_history == [[3, 4, 0]]
        assert new_state.grammar_state[0]._nonterminal_stack == ['q']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.rnn_state[0].hidden_state.cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.rnn_state[0].memory_cell.cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.rnn_state[0].previous_action_embedding.cpu().numpy().tolist(), [4, 4])
        assert_almost_equal(new_state.rnn_state[0].attended_input.cpu().numpy().tolist(), [2, 2])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.rnn_state[0].encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.detach().cpu().numpy(),
                            self.encoder_output_mask.detach().cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions
