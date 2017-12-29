# pylint: disable=invalid-name,no-self-use
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.testing import ModelTestCase
from allennlp.data.semparse.type_declarations import GrammarState
from allennlp.models import WikiTablesSemanticParser
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderState
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderStep


class WikiTablesSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                          "tests/fixtures/data/wikitables/sample_data.examples")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_get_unique_elements(self):
        # pylint: disable=protected-access
        production_rules = [
                # We won't bother with constructing the last element of the ProductionRuleArray
                # here, the Dict[str, torch.Tensor].  It's not necessary for this test.  We'll just
                # give each element a unique index that we can check in the resulting dictionaries.
                # arrays.
                [{"left": ('r', True, 1), "right": ('d', True, 2)},
                 {"left": ('r', True, 1), "right": ('c', True, 3)},
                 {"left": ('d', True, 2), "right": ('entity_1', False, 4)}],
                [{"left": ('r', True, 1), "right": ('d', True, 2)},
                 {"left": ('d', True, 2), "right": ('entity_2', False, 5)},
                 {"left": ('d', True, 2), "right": ('entity_1', False, 4)},
                 {"left": ('d', True, 2), "right": ('entity_3', False, 6)}]
                ]
        nonterminals, terminals = WikiTablesSemanticParser._get_unique_elements(production_rules)
        assert nonterminals == {
                'r': 1,
                'd': 2,
                'c': 3,
                }
        assert terminals == {
                'entity_1': 4,
                'entity_2': 5,
                'entity_3': 6,
                }


class WikiTablesDecoderStepTest(AllenNlpTestCase):
    def test_compute_new_states(self):
        batch_indices = [0, 1, 0]
        action_history = [[1], [3, 4], []]
        score = [Variable(torch.FloatTensor([x])) for x in [.1, 1.1, 2.2]]
        hidden_state = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        memory_cell = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        previous_action_embedding = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        attended_question = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        grammar_state = [GrammarState(['e'], {}, {}, {}) for _ in batch_indices]
        encoder_outputs = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        encoder_output_mask = torch.FloatTensor([[1, 1], [1, 0], [1, 1]])
        action_embeddings = torch.FloatTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        action_indices = {
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
        possible_actions = [
                [{'left': ('e', True, None), 'right': ('f', False, None)},
                 {'left': ('e', True, None), 'right': ('g', True, None)},
                 {'left': ('e', True, None), 'right': ('h', True, None)},
                 {'left': ('e', True, None), 'right': ('i', True, None)},
                 {'left': ('e', True, None), 'right': ('j', True, None)}],
                [{'left': ('e', True, None), 'right': ('q', True, None)},
                 {'left': ('e', True, None), 'right': ('g', True, None)},
                 {'left': ('e', True, None), 'right': ('h', True, None)},
                 {'left': ('e', True, None), 'right': ('i', True, None)}]]
        state = WikiTablesDecoderState(batch_indices=batch_indices,
                                       action_history=action_history,
                                       score=score,
                                       hidden_state=hidden_state,
                                       memory_cell=memory_cell,
                                       previous_action_embedding=previous_action_embedding,
                                       attended_question=attended_question,
                                       grammar_state=grammar_state,
                                       encoder_outputs=encoder_outputs,
                                       encoder_output_mask=encoder_output_mask,
                                       action_embeddings=action_embeddings,
                                       action_indices=action_indices,
                                       possible_actions=possible_actions)
        log_probs = Variable(torch.FloatTensor([[.1, .9, -.1, .2],
                                                [.3, .1, 0, .8],
                                                [.1, .25, .3, .4]]))
        considered_actions = [[0, 1, 2, 3], [0, 3], [0, 2, 4]]
        allowed_actions = [{2, 3}, {0}, {4}]
        max_actions = 1
        step_action_embeddings = torch.FloatTensor([[[1, 1], [0, 0], [2, 2], [3, 3]],
                                                    [[4, 4], [3, 3], [0, 0], [0, 0]],
                                                    [[1, 1], [2, 2], [5, 5], [0, 0]]])
        new_hidden_state = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(batch_indices))]
        new_memory_cell = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(batch_indices))]
        new_attended_question = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(batch_indices))]
        new_states = WikiTablesDecoderStep._compute_new_states(state,
                                                               log_probs,
                                                               new_hidden_state,
                                                               new_memory_cell,
                                                               step_action_embeddings,
                                                               new_attended_question,
                                                               considered_actions,
                                                               allowed_actions,
                                                               max_actions)

        assert len(new_states) == 2
        new_state = new_states[0]
        assert new_state.batch_indices == [0]
        assert new_state.action_history == [[4]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [2.2 + .3])
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [3, 3])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [3, 3])

        # (batch_index, action_index) of (0, 4) maps to action 5 in the global action space, which
        # has embedding [5, 5].
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [5, 5])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [3, 3])
        assert new_state.grammar_state[0]._nonterminal_stack == ['j']
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.cpu().numpy(),
                            encoder_output_mask.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            action_embeddings.cpu().numpy())
        assert new_state.action_indices == action_indices
        assert new_state.possible_actions == possible_actions

        new_state = new_states[1]
        assert new_state.batch_indices == [1]
        assert new_state.action_history == [[3, 4, 0]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [1.1 + .3])
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [2, 2])
        # (batch_index, action_index) of (1, 0) maps to action 4 in the global action space, which
        # has embedding [4, 4].
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [4, 4])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [2, 2])
        assert new_state.grammar_state[0]._nonterminal_stack == ['q']
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.cpu().numpy(),
                            encoder_output_mask.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            action_embeddings.cpu().numpy())
        assert new_state.action_indices == action_indices
        assert new_state.possible_actions == possible_actions
