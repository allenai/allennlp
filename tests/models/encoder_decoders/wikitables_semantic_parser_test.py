# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple

from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.testing import ModelTestCase
from allennlp.common import Params
from allennlp.data.semparse.type_declarations import GrammarState
from allennlp.data.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.models import Model, WikiTablesSemanticParser
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderState
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderStep
from allennlp.modules import SimilarityFunction


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

    def test_embed_actions_works_with_batched_and_padded_input(self):
        # pylint: disable=protected-access
        params = Params.from_file(self.param_file)
        params['model']['embed_terminals'] = True
        model = Model.from_params(self.vocab, params['model'])
        nonterminal_embedding = model._nonterminal_embedder._token_embedders['tokens']
        terminal_encoder = model._terminal_embedder._token_embedders['token_characters']
        start_id = model.vocab.get_token_index(START_SYMBOL, 'rule_labels')
        start_tensor = Variable(torch.LongTensor([start_id]))
        rule2 = model.vocab.get_token_from_index(2, 'rule_labels')
        rule2_tensor = Variable(torch.LongTensor([2]))
        rule3 = model.vocab.get_token_from_index(3, 'rule_labels')
        rule3_tensor = Variable(torch.LongTensor([3]))
        rule4 = model.vocab.get_token_from_index(4, 'rule_labels')
        rule4_tensor = Variable(torch.LongTensor([4]))
        char2 = model.vocab.get_token_from_index(2, 'token_characters')
        char2_tensor = Variable(torch.LongTensor([2, 2, 2]))
        char3 = model.vocab.get_token_from_index(3, 'token_characters')
        char3_tensor = Variable(torch.LongTensor([3, 3, 3]))
        char4 = model.vocab.get_token_from_index(4, 'token_characters')
        char4_tensor = Variable(torch.LongTensor([4, 4, 4]))
        actions = [[{'left': (rule2, True, {'tokens': rule2_tensor}),
                     'right': (char2 * 3, False, {'token_characters': char2_tensor})},
                    {'left': (rule3, True, {'tokens': rule3_tensor}),
                     'right': (char3 * 3, False, {'token_characters': char3_tensor})},
                    # This one is padding; the tensors shouldn't matter here.
                    {'left': ('', True, {'tokens': rule3_tensor}),
                     'right': ('', False, {'token_characters': char3_tensor})}],
                   [{'left': (rule2, True, {'tokens': rule2_tensor}),
                     'right': (char2 * 3, False, {'token_characters': char2_tensor})},
                    {'left': (rule4, True, {'tokens': rule4_tensor}),
                     'right': (char4 * 3, False, {'token_characters': char4_tensor})},
                    {'left': (START_SYMBOL, True, {'tokens': start_tensor}),
                     'right': (rule2, True, {'tokens': rule2_tensor})}]]
        embedded_actions, action_indices, initial_action_embedding = model._embed_actions(actions)
        assert embedded_actions.size(0) == 4
        assert action_indices[(0, 0)] == action_indices[(1, 0)]
        assert len(set(action_indices.values())) == 4

        # Now we'll go through all four unique actions and make sure the embedding is as we expect.
        action_embedding = embedded_actions[action_indices[(0, 0)]]
        left_side_embedding = nonterminal_embedding(rule2_tensor)
        right_side_embedding = terminal_encoder(char2_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0)
        expected_action_embedding = torch.cat([left_side_embedding, right_side_embedding],
                                              dim=-1).squeeze(0)
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(0, 1)]]
        left_side_embedding = nonterminal_embedding(rule3_tensor)
        right_side_embedding = terminal_encoder(char3_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0)
        expected_action_embedding = torch.cat([left_side_embedding, right_side_embedding],
                                              dim=-1).squeeze(0)
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(1, 1)]]
        left_side_embedding = nonterminal_embedding(rule4_tensor)
        right_side_embedding = terminal_encoder(char4_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0)
        expected_action_embedding = torch.cat([left_side_embedding, right_side_embedding],
                                              dim=-1).squeeze(0)
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(1, 2)]]
        left_side_embedding = nonterminal_embedding(start_tensor)
        right_side_embedding = nonterminal_embedding(rule2_tensor)
        expected_action_embedding = torch.cat([left_side_embedding, right_side_embedding],
                                              dim=-1).squeeze(0)
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        # Finally, we'll check that the embedding for the initial action is as we expect.
        start_embedding = nonterminal_embedding(start_tensor).squeeze(0)
        zeros = Variable(start_embedding.data.new(start_embedding.size(-1)).fill_(0).float())
        expected_action_embedding = torch.cat([zeros, start_embedding], dim=-1)
        assert_almost_equal(initial_action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

    def test_embed_actions_does_not_embed_terminals_when_set(self):
        # pylint: disable=protected-access
        model = self.model
        model._embed_terminals = False
        nonterminal_embedding = model._nonterminal_embedder._token_embedders['tokens']
        start_id = model.vocab.get_token_index(START_SYMBOL, 'rule_labels')
        start_tensor = Variable(torch.LongTensor([start_id]))
        rule2 = model.vocab.get_token_from_index(2, 'rule_labels')
        rule2_tensor = Variable(torch.LongTensor([2]))
        rule3 = model.vocab.get_token_from_index(3, 'rule_labels')
        rule3_tensor = Variable(torch.LongTensor([3]))
        rule4 = model.vocab.get_token_from_index(4, 'rule_labels')
        rule4_tensor = Variable(torch.LongTensor([4]))
        char2 = model.vocab.get_token_from_index(2, 'token_characters')
        char2_tensor = Variable(torch.LongTensor([2, 2, 2]))
        char3 = model.vocab.get_token_from_index(3, 'token_characters')
        char3_tensor = Variable(torch.LongTensor([3, 3, 3]))
        char4 = model.vocab.get_token_from_index(4, 'token_characters')
        char4_tensor = Variable(torch.LongTensor([4, 4, 4]))
        actions = [[{'left': (rule2, True, {'tokens': rule2_tensor}),
                     'right': (char2 * 3, False, {'token_characters': char2_tensor})},
                    {'left': (rule3, True, {'tokens': rule3_tensor}),
                     'right': (char3 * 3, False, {'token_characters': char3_tensor})},
                    # This one is padding; the tensors shouldn't matter here.
                    {'left': ('', True, {'tokens': rule3_tensor}),
                     'right': ('', False, {'token_characters': char3_tensor})}],
                   [{'left': (rule2, True, {'tokens': rule2_tensor}),
                     'right': (char2 * 3, False, {'token_characters': char2_tensor})},
                    {'left': (rule4, True, {'tokens': rule4_tensor}),
                     'right': (char4 * 3, False, {'token_characters': char4_tensor})},
                    {'left': (START_SYMBOL, True, {'tokens': start_tensor}),
                     'right': (rule2, True, {'tokens': rule2_tensor})}]]
        embedded_actions, action_indices, _ = model._embed_actions(actions)
        assert embedded_actions.size(0) == 1
        assert action_indices[(1, 2)] == 0  # non-terminals should have lower indices than terminals
        assert action_indices[(0, 0)] == action_indices[(1, 0)]
        assert len(set(action_indices.values())) == 4

        # Now we'll go through all four unique actions and make sure the embedding is as we expect.
        action_embedding = embedded_actions[action_indices[(1, 2)]]
        left_side_embedding = nonterminal_embedding(start_tensor)
        right_side_embedding = nonterminal_embedding(rule2_tensor)
        expected_action_embedding = torch.cat([left_side_embedding, right_side_embedding],
                                              dim=-1).squeeze(0)
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

    def test_map_entity_productions(self):
        # pylint: disable=protected-access
        # (batch_size, num_entities, num_question_tokens) = (3, 4, 5)
        linking_scores = Variable(torch.rand(3, 4, 5))
        # Because we only need a small piece of the WikiTablesWorld and TableKnowledgeGraph, we'll
        # just use some namedtuples to fake the part of the API that we need, instead of going to
        # the trouble of constructing the full objects.
        FakeTable = namedtuple('FakeTable', ['entities'])
        FakeWorld = namedtuple('FakeWorld', ['table_graph'])
        entities = [['fb:cell.2010', 'fb:cell.2011', 'fb:row.row.year', 'fb:row.row.year2'],
                    ['fb:cell.2012', 'fb:cell.2013', 'fb:row.row.year'],
                    ['fb:cell.2010', 'fb:row.row.year']]
        worlds = [FakeWorld(FakeTable(entity_list)) for entity_list in entities]
        # The left-hand side of each action here will not be read, so we won't bother constructing
        # it.  Same with the RHS tensors.  NT* here is just saying "some non-terminal".
        actions = [[{'left': None, 'right': ('NT1', True, None)},
                    {'left': None, 'right': ('NT2', True, None)},
                    {'left': None, 'right': ('NT3', True, None)},
                    {'left': None, 'right': ('fb:cell.2010', True, None)},
                    {'left': None, 'right': ('fb:cell.2011', True, None)},
                    {'left': None, 'right': ('fb:row.row.year', True, None)},
                    {'left': None, 'right': ('fb:row.row.year2', True, None)}],
                   [{'left': None, 'right': ('NT1', True, None)},
                    {'left': None, 'right': ('fb:cell.2012', True, None)},
                    {'left': None, 'right': ('fb:cell.2013', True, None)},
                    {'left': None, 'right': ('fb:row.row.year', True, None)}],
                   [{'left': None, 'right': ('NT4', True, None)},
                    {'left': None, 'right': ('fb:cell.2010', True, None)},
                    {'left': None, 'right': ('fb:row.row.year', True, None)}]]
        flattened_linking_scores, actions_to_entities = \
                WikiTablesSemanticParser._map_entity_productions(linking_scores, worlds, actions)
        assert_almost_equal(flattened_linking_scores.data.cpu().numpy(),
                            linking_scores.view(3 * 4, 5).data.cpu().numpy())
        assert actions_to_entities == {
                (0, 3): 0,
                (0, 4): 1,
                (0, 5): 2,
                (0, 6): 3,
                (1, 1): 4,
                (1, 2): 5,
                (1, 3): 6,
                (2, 1): 8,
                (2, 2): 9,
                }


class WikiTablesDecoderStepTest(AllenNlpTestCase):
    def setUp(self):
        super(WikiTablesDecoderStepTest, self).setUp()

        batch_indices = [0, 1, 0]
        action_history = [[1], [3, 4], []]
        score = [Variable(torch.FloatTensor([x])) for x in [.1, 1.1, 2.2]]
        hidden_state = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        memory_cell = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        previous_action_embedding = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        attended_question = [torch.FloatTensor([i, i]) for i in range(len(batch_indices))]
        grammar_state = [GrammarState(['e'], {}, {}, {}) for _ in batch_indices]
        self.encoder_outputs = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
        self.encoder_output_mask = Variable(torch.FloatTensor([[1, 1], [1, 0], [1, 1]]))
        self.action_embeddings = torch.FloatTensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
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
        self.possible_actions = [
                [{'left': ('e', True, None), 'right': ('f', False, None)},
                 {'left': ('e', True, None), 'right': ('g', True, None)},
                 {'left': ('e', True, None), 'right': ('h', True, None)},
                 {'left': ('e', True, None), 'right': ('i', True, None)},
                 {'left': ('e', True, None), 'right': ('j', True, None)}],
                [{'left': ('e', True, None), 'right': ('q', True, None)},
                 {'left': ('e', True, None), 'right': ('g', True, None)},
                 {'left': ('e', True, None), 'right': ('h', True, None)},
                 {'left': ('e', True, None), 'right': ('i', True, None)}]]

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
                1: 1,
                2: 1,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 1,
                8: 1,
                }
        self.state = WikiTablesDecoderState(batch_indices=batch_indices,
                                            action_history=action_history,
                                            score=score,
                                            hidden_state=hidden_state,
                                            memory_cell=memory_cell,
                                            previous_action_embedding=previous_action_embedding,
                                            attended_question=attended_question,
                                            grammar_state=grammar_state,
                                            encoder_outputs=self.encoder_outputs,
                                            encoder_output_mask=self.encoder_output_mask,
                                            action_embeddings=self.action_embeddings,
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
        self.state.grammar_state[0] = GrammarState(['e'], {}, valid_actions_1, {})
        self.state.grammar_state[1] = GrammarState(['e'], {}, valid_actions_2, {})
        self.state.grammar_state[2] = GrammarState(['e'], {}, valid_actions_3, {})
        self.state.action_embeddings = self.state.action_embeddings[:2]
        considered, to_embed, to_link = WikiTablesDecoderStep._get_actions_to_consider(self.state)
        # These are _global_ action indices.  They come from actions [[(0, 1), (0, 0)], [(1, 1)], []].
        expected_to_embed = [[0, 1], [0], []]
        assert to_embed == expected_to_embed
        # These are _batch_ action indices with a _global_ action index above num_global_actions,
        # sorted by their _global_ action index.
        # They come from actions [[(0, 2), (0, 4)], [(1, 3), (1, 0)], [(0, 3), (0, 4)]].
        expected_to_link = [[2, 4], [3, 0], [2, 3, 4]]
        assert to_link == expected_to_link
        # These are _batch_ action indices, sorted by _global_ action index, with padding in
        # between the embedded actions and the linked actions.
        expected_considered = [[1, 0, 2, 4, -1], [1, -1, 3, 0, -1], [-1, -1, 2, 3, 4]]
        assert considered == expected_considered

    def test_get_actions_to_consider_returns_none_if_no_linked_actions(self):
        # pylint: disable=protected-access
        valid_actions_1 = {'e': [0, 1, 2, 4]}
        valid_actions_2 = {'e': [0, 1, 3]}
        valid_actions_3 = {'e': [2, 3, 4]}
        self.state.grammar_state[0] = GrammarState(['e'], {}, valid_actions_1, {})
        self.state.grammar_state[1] = GrammarState(['e'], {}, valid_actions_2, {})
        self.state.grammar_state[2] = GrammarState(['e'], {}, valid_actions_3, {})
        considered, to_embed, to_link = WikiTablesDecoderStep._get_actions_to_consider(self.state)
        # These are _global_ action indices.  All of the actions in this case are embedded, so this
        # is just a mapping from the valid actions above to their global ids.
        expected_to_embed = [[0, 1, 2, 5], [0, 3, 4], [2, 3, 5]]
        assert to_embed == expected_to_embed
        # There are no linked actions (all of them are embedded), so this should be None.
        assert to_link is None
        # These are _batch_ action indices, sorted by _global_ action index, with padding in
        # between the embedded actions and the linked actions.  Because there are no linked
        # actions, this is basically just the valid_actions for each group element padded with -1s,
        # and sorted by global action index.
        expected_considered = [[1, 0, 2, 4], [1, 3, 0, -1], [2, 3, 4, -1]]
        assert considered == expected_considered

    def test_get_action_embeddings(self):
        action_embeddings = Variable(torch.rand(5, 4))
        self.state.action_embeddings = action_embeddings
        actions_to_embed = [[0, 4], [1], [2, 3, 4]]
        embeddings, mask = WikiTablesDecoderStep._get_action_embeddings(self.state, actions_to_embed)
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
        decoder_step = WikiTablesDecoderStep(1, 5, SimilarityFunction.from_params(Params({})), 2)
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
        assert_almost_equal(type_embeddings[0, 0].data.cpu().numpy(), embedding_matrix[1])
        assert_almost_equal(type_embeddings[0, 1].data.cpu().numpy(), embedding_matrix[1])
        assert_almost_equal(type_embeddings[1, 0].data.cpu().numpy(), embedding_matrix[0])
        assert_almost_equal(type_embeddings[1, 1].data.cpu().numpy(), embedding_matrix[1])
        assert_almost_equal(type_embeddings[1, 2].data.cpu().numpy(), embedding_matrix[1])
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
        new_hidden_state = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(allowed_actions))]
        new_memory_cell = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(allowed_actions))]
        new_attended_question = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(allowed_actions))]
        new_states = WikiTablesDecoderStep._compute_new_states(self.state,
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
        # For batch instance 0, we should have selected action 4 from group index 2.
        assert new_state.batch_indices == [0]
        # These three have values taken from what's defined in setUp() - the prior action history
        # (empty in this case), the initial score (2.2), and the nonterminals corresponding to the
        # action we picked ('j').
        assert new_state.action_history == [[4]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [2.2 + .3])
        assert new_state.grammar_state[0]._nonterminal_stack == ['j']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [3, 3])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [3, 3])
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [5, 5])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [3, 3])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 1, we should have selected action 0 from group index 1.
        assert new_state.batch_indices == [1]
        # These three have values taken from what's defined in setUp() - the prior action history
        # ([3, 4]), the initial score (1.1), and the nonterminals corresponding to the action we
        # picked ('q').
        assert new_state.action_history == [[3, 4, 0]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [1.1 + .3])
        assert new_state.grammar_state[0]._nonterminal_stack == ['q']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [4, 4])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [2, 2])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.data.cpu().numpy(),
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
        new_hidden_state = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(considered_actions))]
        new_memory_cell = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(considered_actions))]
        new_attended_question = [torch.FloatTensor([i + 1, i + 1]) for i in range(len(considered_actions))]
        new_states = WikiTablesDecoderStep._compute_new_states(self.state,
                                                               log_probs,
                                                               new_hidden_state,
                                                               new_memory_cell,
                                                               step_action_embeddings,
                                                               new_attended_question,
                                                               considered_actions,
                                                               allowed_actions=None,
                                                               max_actions=max_actions)

        assert len(new_states) == 2
        new_state = new_states[0]
        # For batch instance 0, we should have selected action 1 from group index 0.
        assert new_state.batch_indices == [0]
        # These three have values taken from what's defined in setUp() - the prior action history
        # ([1]), the initial score (0.1), and the nonterminals corresponding to the
        # action we picked ('j').
        assert new_state.action_history == [[1, 1]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [0.1 + .9])
        assert new_state.grammar_state[0]._nonterminal_stack == ['g']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [1, 1])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [1, 1])
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [9, 9])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [1, 1])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions

        new_state = new_states[1]
        # For batch instance 0, we should have selected action 0 from group index 1.
        assert new_state.batch_indices == [1]
        # These three have values taken from what's defined in setUp() - the prior action history
        # ([3, 4]), the initial score (1.1), and the nonterminals corresponding to the action we
        # picked ('q').
        assert new_state.action_history == [[3, 4, 0]]
        assert_almost_equal(new_state.score[0].data.cpu().numpy().tolist(), [1.1 + .3])
        assert new_state.grammar_state[0]._nonterminal_stack == ['q']
        # All of these values come from the objects instantiated directly above.
        assert_almost_equal(new_state.hidden_state[0].cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.memory_cell[0].cpu().numpy().tolist(), [2, 2])
        assert_almost_equal(new_state.previous_action_embedding[0].cpu().numpy().tolist(), [4, 4])
        assert_almost_equal(new_state.attended_question[0].cpu().numpy().tolist(), [2, 2])
        # And these should just be copied from the prior state.
        assert_almost_equal(new_state.encoder_outputs.cpu().numpy(),
                            self.encoder_outputs.cpu().numpy())
        assert_almost_equal(new_state.encoder_output_mask.data.cpu().numpy(),
                            self.encoder_output_mask.data.cpu().numpy())
        assert_almost_equal(new_state.action_embeddings.cpu().numpy(),
                            self.action_embeddings.cpu().numpy())
        assert new_state.action_indices == self.action_indices
        assert new_state.possible_actions == self.possible_actions
