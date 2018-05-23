# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple
import os
import shutil

from flaky import flaky
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.models import Model, WikiTablesMmlSemanticParser
from allennlp.training.metrics.wikitables_accuracy import SEMPRE_DIR

class WikiTablesMmlSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_dir = not os.path.exists(SEMPRE_DIR)
        super(WikiTablesMmlSemanticParserTest, self).setUp()
        self.set_up_model(f"tests/fixtures/semantic_parsing/wikitables/experiment.json",
                          "tests/fixtures/data/wikitables/sample_data.examples")

    def tearDown(self):
        super().tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_dir and os.path.exists(SEMPRE_DIR):
            shutil.rmtree('data')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_elmo_mixture_no_features_model_can_train_save_and_load(self):
        param_file = 'tests/fixtures/semantic_parsing/wikitables/experiment-mixture.json'
        self.ensure_model_can_train_save_and_load(param_file)

    @flaky
    def test_elmo_no_features_can_train_save_and_load(self):
        param_file = 'tests/fixtures/semantic_parsing/wikitables/experiment-elmo-no-features.json'
        self.ensure_model_can_train_save_and_load(param_file, tolerance=1e-2)

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

    def test_embed_actions_works_with_batched_and_padded_input(self):
        params = Params.from_file(self.param_file)
        model = Model.from_params(self.vocab, params['model'])
        action_embedding_weights = model._action_embedder.weight
        rule1 = model.vocab.get_token_from_index(1, 'rule_labels')
        rule1_tensor = Variable(torch.LongTensor([1]))
        rule2 = model.vocab.get_token_from_index(2, 'rule_labels')
        rule2_tensor = Variable(torch.LongTensor([2]))
        rule3 = model.vocab.get_token_from_index(3, 'rule_labels')
        rule3_tensor = Variable(torch.LongTensor([3]))
        actions = [[(rule1, True, rule1_tensor),
                    (rule2, True, rule2_tensor),
                    # This one is padding; the tensors shouldn't matter here.
                    ('', False, None)],
                   [(rule3, True, rule3_tensor),
                    ('instance_action', False, None),
                    (rule1, True, rule1_tensor)]]

        embedded_actions, _, _, action_indices = model._embed_actions(actions)
        assert action_indices[(0, 0)] == action_indices[(1, 2)]
        assert action_indices[(1, 1)] == -1
        assert len(set(action_indices.values())) == 4

        # Now we'll go through all three unique actions and make sure the embedding is as we expect.
        action_embedding = embedded_actions[action_indices[(0, 0)]]
        expected_action_embedding = action_embedding_weights[action_indices[(0, 0)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(0, 1)]]
        expected_action_embedding = action_embedding_weights[action_indices[(0, 1)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

        action_embedding = embedded_actions[action_indices[(1, 0)]]
        expected_action_embedding = action_embedding_weights[action_indices[(1, 0)]]
        assert_almost_equal(action_embedding.cpu().data.numpy(),
                            expected_action_embedding.cpu().data.numpy())

    def test_map_entity_productions(self):
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
        # The tensors here for the global actions won't actually be read, so we're not constructing
        # them.
        # it.  Same with the RHS tensors.  NT* here is just saying "some non-terminal".
        actions = [[('@START@ -> r', True, None),
                    ('@START@ -> c', True, None),
                    ('@START@ -> <c,r>', True, None),
                    ('c -> fb:cell.2010', False, None),
                    ('c -> fb:cell.2011', False, None),
                    ('<c,r> -> fb:row.row.year', False, None),
                    ('<c,r> -> fb:row.row.year2', False, None)],
                   [('@START@ -> c', True, None),
                    ('c -> fb:cell.2012', False, None),
                    ('c -> fb:cell.2013', False, None),
                    ('<c,r> -> fb:row.row.year', False, None)],
                   [('@START@ -> c', True, None),
                    ('c -> fb:cell.2010', False, None),
                    ('<c,r> -> fb:row.row.year', False, None)]]
        flattened_linking_scores, actions_to_entities = \
                WikiTablesMmlSemanticParser._map_entity_productions(linking_scores, worlds, actions)
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
