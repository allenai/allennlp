# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple

from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.models.semantic_parsing.ratecalculus.ratecalculus_semantic_parser import RateCalculusSemanticParser

class RateCalculusSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(RateCalculusSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "ratecalculus" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "ratecalculus" / "sample_data.json"))

    #@flaky
    #def test_model_can_train_save_and_load(self):
    #    self.ensure_model_can_train_save_and_load(self.param_file)

    def test_get_neighbor_indices(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = Variable(torch.LongTensor([]))

        neighbor_indices = self.model._get_neighbor_indices(worlds, num_entities, tensor)

        # Checks for the correct shape meaning dimension 2 has size num_neighbors,
        # padding of -1 is used, and correct neighbor indices.
        assert_almost_equal(neighbor_indices.data.numpy(), [[[-1],
                                                             [2],
                                                             [1]]])

    def test_get_type_vector(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = Variable(torch.LongTensor([]))
        type_vector, _ = self.model._get_type_vector(worlds, num_entities, tensor)
        # Verify that both types are present and padding used for non existent entities.
        assert_almost_equal(type_vector.data.numpy(), [[[1, 0, 0, 0],
                                                        [1, 0, 0, 0],
                                                        [1, 0, 0, 0]]])

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
        true_probability = [[[0.02788339, 0.56005281, 0.20603192, 0., 0.],
                             [0.26714143, 0.00179998, 0.72616571, 0., 0.]],
                            [[0.02788339, 0.56005281, 0.20603192, 0., 0.],
                             [0., 0., 0., 0., 0.]]]
        assert_almost_equal(entity_probability.data.cpu().numpy(), true_probability)

    def get_fake_worlds(self):
        # Generate a toy RateCalculusWorld.
        FakeQuestionKnowledgeGraph = namedtuple('FakeQuestionKnowledgeGraph', ['entities', 'neighbors'])
        FakeWorld = namedtuple('FakeWorld', ['question_knowledge_graph'])
        entities = [['1', 's', 'p']]
        neighbors = [{'1': [],
                      's': ['p'],
                      'p': ['s'],
                     }]

        worlds = [FakeWorld(FakeQuestionKnowledgeGraph(entity_list, entity2neighbors))
                  for entity_list, entity2neighbors in zip(entities, neighbors)]
        num_entities = max([len(entity_list) for entity_list in entities])
        return worlds, num_entities

    def test_map_entity_productions(self):
        # (batch_size, num_entities, num_question_tokens) = (3, 4, 5)
        linking_scores = torch.rand(3, 4, 5)
        # Because we only need a small piece of the RateCalculusWorld and KnowledgeGraph, we'll
        # just use some namedtuples to fake the part of the API that we need, instead of going to
        # the trouble of constructing the full objects.
        FakeQuestionKnowledgeGraph = namedtuple('FakeQuestionKnowledgeGraph', ['entities'])
        FakeWorld = namedtuple('FakeWorld', ['question_knowledge_graph'])
        entities = ['1', 's', 'p']
        worlds = [FakeWorld(FakeQuestionKnowledgeGraph(entity_list)) for entity_list in entities]
        # The tensors here for the global actions won't actually be read, so we're not constructing
        # them.
        # it.  Same with the RHS tensors.  NT* here is just saying "some non-terminal".
        actions = [[('@start@ -> b', True, None),
                    ('n -> 1', True, None),
                    ('d -> Dollar', True, None),
                    ('d -> Unit', True, None)]]
        flattened_linking_scores, actions_to_entities = \
                RateCalculusSemanticParser._map_entity_productions(linking_scores, worlds, actions)
        assert_almost_equal(flattened_linking_scores.data.cpu().numpy(),
                            linking_scores.view(3 * 4, 5).data.cpu().numpy())
        assert actions_to_entities == {(0, 1): 0}
