# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple

from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model, WikiTablesSemanticParser
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL

class WikiTablesSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                          "tests/fixtures/data/wikitables/sample_data.examples")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_mixture_feedforward_model_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load('tests/fixtures/encoder_decoder/wikitables_semantic_parser_with_mixture_feedforward/experiment.json')

    def test_model_no_features_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load("tests/fixtures/encoder_decoder/wikitables_semantic_parser_no_features/experiment.json")

    def test_get_neighbor_indices(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = Variable(torch.LongTensor([]))

        neighbor_indices = self.model._get_neighbor_indices(worlds, num_entities, tensor)

        # Checks for the correct shape meaning dimension 2 has size num_neighbors,
        # padding of -1 is used, and correct neighbor indices.
        assert_almost_equal(neighbor_indices.data.numpy(), [[[-1, -1],
                                                             [3, 4],
                                                             [3, 4],
                                                             [1, 2],
                                                             [1, 2]],
                                                            [[-1, -1],
                                                             [2, -1],
                                                             [1, -1],
                                                             [-1, -1],
                                                             [-1, -1]]])

    def test_get_type_vector(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = Variable(torch.LongTensor([]))
        type_vector, _ = self.model._get_type_vector(worlds, num_entities, tensor)
        # Verify that both types are present and padding used for non existent entities.
        assert_almost_equal(type_vector.data.numpy(), [[[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 1],
                                                        [0, 0, 0, 1]],
                                                       [[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 1],
                                                        [0, 0, 0, 0],
                                                        [0, 0, 0, 0]]])

    def test_get_linking_probabilities(self):
        worlds, num_entities = self.get_fake_worlds()
        # (batch_size, num_question_tokens, num_entities)
        linking_scores = [[[-2, 1, 0, -3, 2],
                           [4, -1, 5, -3, 4]],
                          [[0, 1, 8, 10, 10],
                           [3, 2, -1, -2, 1]]]
        linking_scores = Variable(torch.FloatTensor(linking_scores))
        question_mask = Variable(torch.LongTensor([[1, 1], [1, 0]]))
        _, entity_type_dict = self.model._get_type_vector(worlds, num_entities, linking_scores)

        # (batch_size, num_question_tokens, num_entities)
        entity_probability = self.model._get_linking_probabilities(worlds, linking_scores, question_mask,
                                                                   entity_type_dict)

        # The following properties in entity_probability are tested for by true_probability:
        # (1) It has all 0.0 probabilities when there is no question token, as seen for the
        #     second word in the second batch.
        # (2) It has 0.0 probabilities when an entity is masked, as seen in the last two entities
        #     for the second batch instance.
        # (3) The probabilities for entities of the same type with the same question token should
        #     sum to at most 1, but not necessarily 1, because some probability mass goes to the
        #     null entity.  We have three entity types here, so each row should sum to at most 3,
        #     and that number will approach 3 as the unnormalized linking scores for each entity
        #     get higher.
        true_probability = [[[0.1192029, 0.5761169, 0.2119416, 0.0058998, 0.8756006],
                             [0.9820138, 0.0024561, 0.9908675, 0.0008947, 0.9811352]],
                            [[0.5, 0.7310586, 0.9996647, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0]]]
        assert_almost_equal(entity_probability.data.cpu().numpy(), true_probability)

    def get_fake_worlds(self):
        # Generate a toy WikitablesWorld.
        FakeTable = namedtuple('FakeTable', ['entities', 'neighbors'])
        FakeWorld = namedtuple('FakeWorld', ['table_graph'])
        entities = [['0', 'fb:cell.2010', 'fb:cell.2011', 'fb:row.row.year', 'fb:row.row.year2'],
                    ['1', 'fb:cell.2012', 'fb:row.row.year']]
        neighbors = [{'fb:cell.2010': ['fb:row.row.year', 'fb:row.row.year2'],
                      'fb:cell.2011': ['fb:row.row.year', 'fb:row.row.year2'],
                      'fb:row.row.year': ['fb:cell.2010', 'fb:cell.2011'],
                      'fb:row.row.year2': ['fb:cell.2010', 'fb:cell.2011'],
                      '0': [],
                     },
                     {'fb:cell.2012': ['fb:row.row.year'],
                      'fb:row.row.year': ['fb:cell.2012'],
                      '1': [],
                     }]

        worlds = [FakeWorld(FakeTable(entity_list, entity2neighbors))
                  for entity_list, entity2neighbors in zip(entities, neighbors)]
        num_entities = max([len(entity_list) for entity_list in entities])
        return worlds, num_entities

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
