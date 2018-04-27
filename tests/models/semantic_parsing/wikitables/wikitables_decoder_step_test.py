# pylint: disable=invalid-name,no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

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

        batch_indices = [0, 1, 0]
        action_history = [[1], [3, 4], []]
        score = [Variable(torch.FloatTensor([x])) for x in [.1, 1.1, 2.2]]
        hidden_state = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        memory_cell = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        previous_action_embedding = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        attended_question = torch.FloatTensor([[i, i] for i in range(len(batch_indices))])
        grammar_state = [GrammarState(['e'], {}, {}, {}, is_nonterminal) for _ in batch_indices]
        self.encoder_outputs = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        self.encoder_output_mask = Variable(torch.FloatTensor([[1, 1], [1, 0], [1, 1]]))
        self.action_embeddings = torch.FloatTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        self.output_action_embeddings = torch.FloatTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        self.action_biases = torch.FloatTensor([[0], [1], [2], [3], [4], [5]])
        self.action_indices = {
                (0, 0): 1,
                (0, 1): 0,
                (0, 2): 2,
                (0, 3): 3,
                (0, 4): 5,
                (1, 0): 4,
                (1, 1): 0,
                (1, 2): 2,
                (1, 3): 3,
                }
        self.possible_actions = [[('e -> f', False, None),
                                  ('e -> g', True, None),
                                  ('e -> h', True, None),
                                  ('e -> i', True, None),
                                  ('e -> j', True, None)],
                                 [('e -> q', True, None),
                                  ('e -> g', True, None),
                                  ('e -> h', True, None),
                                  ('e -> i', True, None)]]

        # (batch_size, num_entities, num_question_tokens) = (2, 5, 3)
        linking_scores = Variable(torch.Tensor([[[.1, .2, .3],
                                                 [.4, .5, .6],
                                                 [.7, .8, .9],
                                                 [1.0, 1.1, 1.2],
                                                 [1.3, 1.4, 1.5]],
                                                [[-.1, -.2, -.3],
                                                 [-.4, -.5, -.6],
                                                 [-.7, -.8, -.9],
                                                 [-1.0, -1.1, -1.2],
                                                 [-1.3, -1.4, -1.5]]]))
        flattened_linking_scores = linking_scores.view(2 * 5, 3)

        # Maps (batch_index, action_index) to indices into the flattened linking score tensor,
        # which has shae (batch_size * num_entities, num_question_tokens).
        actions_to_entities = {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 2,
                (0, 6): 3,
                (1, 3): 6,
                (1, 4): 7,
                (1, 5): 8,
                }
        entity_types = {
                0: 0,
                1: 2,
                2: 1,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 1,
                8: 2,
                }
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
                                            action_embeddings=self.action_embeddings,
                                            output_action_embeddings=self.output_action_embeddings,
                                            action_biases=self.action_biases,
                                            action_indices=self.action_indices,
                                            possible_actions=self.possible_actions,
                                            flattened_linking_scores=flattened_linking_scores,
                                            actions_to_entities=actions_to_entities,
                                            entity_types=entity_types)

    def test_get_actions_to_consider(self):
        # pylint: disable=protected-access
        valid_actions_1 = {'e': [0, 1, 2, 4]}
        valid_actions_2 = {'e': [0, 1, 3]}
        valid_actions_3 = {'e': [2, 3, 4]}
        self.state.grammar_state[0] = GrammarState(['e'], {}, valid_actions_1, {}, is_nonterminal)
        self.state.grammar_state[1] = GrammarState(['e'], {}, valid_actions_2, {}, is_nonterminal)
        self.state.grammar_state[2] = GrammarState(['e'], {}, valid_actions_3, {}, is_nonterminal)

        # We're making a bunch of the actions linked actions here, pretending that there are only
        # two global actions.
        self.state.action_indices = {
                (0, 0): 1,
                (0, 1): 0,
                (0, 2): -1,
                (0, 3): -1,
                (0, 4): -1,
                (1, 0): -1,
                (1, 1): 0,
                (1, 2): -1,
                (1, 3): -1,
                }

        considered, to_embed, to_link = WikiTablesDecoderStep._get_actions_to_consider(self.state)
        # These are _global_ action indices.  They come from actions [[(0, 0), (0, 1)], [(1, 1)], []].
        expected_to_embed = [[1, 0], [0], []]
        assert to_embed == expected_to_embed
        # These are _batch_ action indices with a _global_ action index of -1.
        # They come from actions [[(0, 2), (0, 4)], [(1, 0), (1, 3)], [(0, 2), (0, 3), (0, 4)]].
        expected_to_link = [[2, 4], [0, 3], [2, 3, 4]]
        assert to_link == expected_to_link
        # These are _batch_ action indices, with padding in between the embedded actions and the
        # linked actions (and after the linked actions, if necessary).
        expected_considered = [[0, 1, 2, 4, -1], [1, -1, 0, 3, -1], [-1, -1, 2, 3, 4]]
        assert considered == expected_considered

    def test_get_actions_to_consider_returns_none_if_no_linked_actions(self):
        # pylint: disable=protected-access
        valid_actions_1 = {'e': [0, 1, 2, 4]}
        valid_actions_2 = {'e': [0, 1, 3]}
        valid_actions_3 = {'e': [2, 3, 4]}
        self.state.grammar_state[0] = GrammarState(['e'], {}, valid_actions_1, {}, is_nonterminal)
        self.state.grammar_state[1] = GrammarState(['e'], {}, valid_actions_2, {}, is_nonterminal)
        self.state.grammar_state[2] = GrammarState(['e'], {}, valid_actions_3, {}, is_nonterminal)
        considered, to_embed, to_link = WikiTablesDecoderStep._get_actions_to_consider(self.state)
        # These are _global_ action indices.  All of the actions in this case are embedded, so this
        # is just a mapping from the valid actions above to their global ids.
        expected_to_embed = [[1, 0, 2, 5], [4, 0, 3], [2, 3, 5]]
        assert to_embed == expected_to_embed
        # There are no linked actions (all of them are embedded), so this should be None.
        assert to_link is None
        # These are _batch_ action indices, with padding in between the embedded actions and the
        # linked actions.  Because there are no linked actions, this is basically just the
        # valid_actions for each group element padded with -1s.
        expected_considered = [[0, 1, 2, 4], [0, 1, 3, -1], [2, 3, 4, -1]]
        assert considered == expected_considered

    def test_get_action_embeddings(self):
        action_embeddings = Variable(torch.rand(5, 4))
        self.state.action_embeddings = action_embeddings
        self.state.output_action_embeddings = action_embeddings
        self.state.action_biases = Variable(torch.rand(5, 1))
        actions_to_embed = [[0, 4], [1], [2, 3, 4]]
        embeddings, _, _, mask = WikiTablesDecoderStep._get_action_embeddings(self.state, actions_to_embed)
        assert_almost_equal(mask.data.cpu().numpy(), [[1, 1, 0], [1, 0, 0], [1, 1, 1]])
        assert tuple(embeddings.size()) == (3, 3, 4)
        assert_almost_equal(embeddings[0, 0].data.cpu().numpy(), action_embeddings[0].data.cpu().numpy())
        assert_almost_equal(embeddings[0, 1].data.cpu().numpy(), action_embeddings[4].data.cpu().numpy())
        assert_almost_equal(embeddings[0, 2].data.cpu().numpy(), action_embeddings[0].data.cpu().numpy())
        assert_almost_equal(embeddings[1, 0].data.cpu().numpy(), action_embeddings[1].data.cpu().numpy())
        assert_almost_equal(embeddings[1, 1].data.cpu().numpy(), action_embeddings[0].data.cpu().numpy())
        assert_almost_equal(embeddings[1, 2].data.cpu().numpy(), action_embeddings[0].data.cpu().numpy())
        assert_almost_equal(embeddings[2, 0].data.cpu().numpy(), action_embeddings[2].data.cpu().numpy())
        assert_almost_equal(embeddings[2, 1].data.cpu().numpy(), action_embeddings[3].data.cpu().numpy())
        assert_almost_equal(embeddings[2, 2].data.cpu().numpy(), action_embeddings[4].data.cpu().numpy())

    def test_get_entity_action_logits(self):
        decoder_step = WikiTablesDecoderStep(1, 5, SimilarityFunction.from_params(Params({})), 5, 3)
        actions_to_link = [[1, 2], [3, 4, 5], [6]]
        # (group_size, num_question_tokens) = (3, 3)
        attention_weights = Variable(torch.Tensor([[.2, .8, 0],
                                                   [.7, .1, .2],
                                                   [.3, .3, .4]]))
        action_logits, mask, type_embeddings = decoder_step._get_entity_action_logits(self.state,
                                                                                      actions_to_link,
                                                                                      attention_weights)
        assert_almost_equal(mask.data.cpu().numpy(), [[1, 1, 0], [1, 1, 1], [1, 0, 0]])

        assert tuple(action_logits.size()) == (3, 3)
        assert_almost_equal(action_logits[0, 0].data.cpu().numpy(), .4 * .2 + .5 * .8 + .6 * 0)
        assert_almost_equal(action_logits[0, 1].data.cpu().numpy(), .7 * .2 + .8 * .8 + .9 * 0)
        assert_almost_equal(action_logits[1, 0].data.cpu().numpy(), -.4 * .7 + -.5 * .1 + -.6 * .2)
        assert_almost_equal(action_logits[1, 1].data.cpu().numpy(), -.7 * .7 + -.8 * .1 + -.9 * .2)
        assert_almost_equal(action_logits[1, 2].data.cpu().numpy(), -1.0 * .7 + -1.1 * .1 + -1.2 * .2)
        assert_almost_equal(action_logits[2, 0].data.cpu().numpy(), 1.0 * .3 + 1.1 * .3 + 1.2 * .4)

        embedding_matrix = decoder_step._entity_type_embedding.weight.data.cpu().numpy()
        assert_almost_equal(type_embeddings[0, 0].data.cpu().numpy(), embedding_matrix[2])
        assert_almost_equal(type_embeddings[0, 1].data.cpu().numpy(), embedding_matrix[1])
        assert_almost_equal(type_embeddings[1, 0].data.cpu().numpy(), embedding_matrix[0])
        assert_almost_equal(type_embeddings[1, 1].data.cpu().numpy(), embedding_matrix[1])
        assert_almost_equal(type_embeddings[1, 2].data.cpu().numpy(), embedding_matrix[2])
        assert_almost_equal(type_embeddings[2, 0].data.cpu().numpy(), embedding_matrix[0])

    def test_compute_new_states(self):
        # pylint: disable=protected-access
        log_probs = Variable(torch.FloatTensor([[.1, .9, -.1, .2],
                                                [.3, 1.1, .1, .8],
                                                [.1, .25, .3, .4]]))
        considered_actions = [[0, 1, 2, 3], [0, -1, 3, -1], [0, 2, 4, -1]]
        allowed_actions = [{2, 3}, {0}, {4}]
        max_actions = 1
        step_action_embeddings = torch.FloatTensor([[[1, 1], [9, 9], [2, 2], [3, 3]],
                                                    [[4, 4], [9, 9], [3, 3], [9, 9]],
                                                    [[1, 1], [2, 2], [5, 5], [9, 9]]])
        new_hidden_state = torch.FloatTensor([[i + 1, i + 1] for i in range(len(allowed_actions))])
        new_memory_cell = torch.FloatTensor([[i + 1, i + 1] for i in range(len(allowed_actions))])
        new_attended_question = torch.FloatTensor([[i + 1, i + 1] for i in range(len(allowed_actions))])
        new_attention_weights = torch.FloatTensor([[i + 1, i + 1] for i in range(len(allowed_actions))])
        new_states = WikiTablesDecoderStep._compute_new_states(self.state,
                                                               log_probs,
                                                               new_hidden_state,
                                                               new_memory_cell,
                                                               step_action_embeddings,
                                                               new_attended_question,
                                                               new_attention_weights,
                                                               considered_actions,
                                                               allowed_actions,
                                                               max_actions)

        assert len(new_states) == 2
        new_state = new_states[0]
        # For batch instance 0, we should have selected action 4 from group index 2.
        assert new_state.batch_indices == [0]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [.3])
        # These have values taken from what's defined in setUp() - the prior action history
        # (empty in this case)  and the nonterminals corresponding to the action we picked ('j').
        assert new_state.action_history == [[4]]
        assert new_state.grammar_state[0]._nonterminal_stack == ['j']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.rnn_state[0].hidden_state.cpu().numpy().tolist(), [3, 3])
        assert_almost_equal(new_state.rnn_state[0].memory_cell.cpu().numpy().tolist(), [3, 3])
        assert_almost_equal(new_state.rnn_state[0].previous_action_embedding.cpu().numpy().tolist(), [5, 5])
        assert_almost_equal(new_state.rnn_state[0].attended_input.cpu().numpy().tolist(), [3, 3])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.rnn_state[0].encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 1, we should have selected action 0 from group index 1.
        assert new_state.batch_indices == [1]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [.3])
        # These two have values taken from what's defined in setUp() - the prior action history
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
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

    def test_compute_new_states_with_no_action_constraints(self):
        # pylint: disable=protected-access
        # This test is basically identical to the previous one, but without specifying
        # `allowed_actions`.  This makes sure we get the right behavior at test time.
        log_probs = Variable(torch.FloatTensor([[.1, .9, -.1, .2],
                                                [.3, 1.1, .1, .8],
                                                [.1, .25, .3, .4]]))
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
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [.9])
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
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 0, we should have selected action 0 from group index 1.
        assert new_state.batch_indices == [1]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [.3])
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
        assert_almost_equal(new_state.rnn_state[0].encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions
