# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple
import os
import pytest

from flaky import flaky
from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.training.metrics.wikitables_accuracy import SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH

@pytest.mark.java
class WikiTablesMmlSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

        # The model tests are run with respect to the module root, so check if abbreviations
        # and grammar already exist there (since we want to clean up module root after test)
        self.module_root_abbreviations_path = self.MODULE_ROOT / "data" / "abbreviations.tsv"
        self.module_root_grammar_path = self.MODULE_ROOT / "data" / "grow.grammar"
        self.should_remove_root_sempre_abbreviations = not os.path.exists(self.module_root_abbreviations_path)
        self.should_remove_root_sempre_grammar = not os.path.exists(self.module_root_grammar_path)

        super(WikiTablesMmlSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "wikitables" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "wikitables" / "sample_data.examples"))

    def tearDown(self):
        super().tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)
        if self.should_remove_root_sempre_abbreviations and os.path.exists(self.module_root_abbreviations_path):
            os.remove(self.module_root_abbreviations_path)
        if self.should_remove_root_sempre_grammar and os.path.exists(self.module_root_grammar_path):
            os.remove(self.module_root_grammar_path)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_mixture_no_features_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'experiment-mixture.json'
        self.ensure_model_can_train_save_and_load(param_file)

    @flaky
    def test_elmo_no_features_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'experiment-elmo-no-features.json'
        self.ensure_model_can_train_save_and_load(param_file, tolerance=1e-2)

    def test_get_neighbor_indices(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = torch.LongTensor([])

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
        tensor = torch.LongTensor([])
        type_vector, _ = self.model._get_type_vector(worlds, num_entities, tensor)
        # Verify that both types are present and padding used for non existent entities.
        assert_almost_equal(type_vector.data.numpy(), [[0, 1, 1, 3, 3],
                                                       [0, 1, 3, 0, 0]])

    def test_get_linking_probabilities(self):
        worlds, num_entities = self.get_fake_worlds()
        # (batch_size, num_question_tokens, num_entities)
        linking_scores = [[[-2, 1, 0, -3, 2],
                           [4, -1, 5, -3, 4]],
                          [[0, 1, 8, 10, 10],
                           [3, 2, -1, -2, 1]]]
        linking_scores = torch.FloatTensor(linking_scores)
        question_mask = torch.LongTensor([[1, 1], [1, 0]])
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
        assert_almost_equal(entity_probability.detach().cpu().numpy(), true_probability)

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
