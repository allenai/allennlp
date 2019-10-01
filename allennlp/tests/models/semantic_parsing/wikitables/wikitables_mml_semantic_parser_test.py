from collections import namedtuple
from flaky import flaky
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.iterators import DataIterator


class WikiTablesMmlSemanticParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        config_path = self.FIXTURES_ROOT / "semantic_parsing" / "wikitables" / "experiment.json"
        data_path = self.FIXTURES_ROOT / "data" / "wikitables" / "sample_data.examples"
        self.set_up_model(config_path, data_path)

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_model_decode(self):
        params = Params.from_file(self.param_file)
        iterator_params = params["iterator"]
        iterator = DataIterator.from_params(iterator_params)
        iterator.index_with(self.model.vocab)
        model_batch = next(iterator(self.dataset, shuffle=False))
        self.model.training = False
        forward_output = self.model(**model_batch)
        decode_output = self.model.decode(forward_output)
        assert "predicted_actions" in decode_output

    def test_get_neighbor_indices(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = torch.LongTensor([])

        neighbor_indices = self.model._get_neighbor_indices(worlds, num_entities, tensor)

        # Checks for the correct shape meaning dimension 2 has size num_neighbors,
        # padding of -1 is used, and correct neighbor indices.
        assert_almost_equal(
            neighbor_indices.data.numpy(),
            [
                [[-1, -1], [4, -1], [4, -1], [5, -1], [1, 2], [3, -1]],
                [[-1, -1], [2, -1], [1, -1], [-1, -1], [-1, -1], [-1, -1]],
            ],
        )

    def test_get_type_vector(self):
        worlds, num_entities = self.get_fake_worlds()
        tensor = torch.LongTensor([])
        type_vector, _ = self.model._get_type_vector(worlds, num_entities, tensor)
        # Verify that the appropriate types are present and padding used for non existent entities.
        assert_almost_equal(type_vector.data.numpy(), [[0, 0, 0, 3, 1, 4], [0, 0, 1, 0, 0, 0]])

    def test_get_linking_probabilities(self):
        worlds, num_entities = self.get_fake_worlds()
        # (batch_size, num_question_tokens, num_entities)
        linking_scores = [
            [[-2, 1, 0, -3, 2, -2], [4, -1, 5, -3, 4, 3]],
            [[0, 1, 8, 10, 10, 4], [3, 2, -1, -2, 1, -6]],
        ]
        linking_scores = torch.FloatTensor(linking_scores)
        question_mask = torch.LongTensor([[1, 1], [1, 0]])
        _, entity_type_dict = self.model._get_type_vector(worlds, num_entities, linking_scores)

        # (batch_size, num_question_tokens, num_entities)
        entity_probability = self.model._get_linking_probabilities(
            worlds, linking_scores, question_mask, entity_type_dict
        )

        # The following properties in entity_probability are tested for by true_probability:
        # (1) It has all 0.0 probabilities when there is no question token, as seen for the
        #     second word in the second batch.
        # (2) It has 0.0 probabilities when an entity is masked, as seen in the last three entities
        #     for the second batch instance.
        # (3) The probabilities for entities of the same type with the same question token should
        #     sum to at most 1, but not necessarily 1, because some probability mass goes to the
        #     null entity.  We have four entity types here, so each row should sum to at most 4,
        #     and that number will approach 4 as the unnormalized linking scores for each entity
        #     get higher.
        true_probability = [
            [
                [0.02788338, 0.56005275, 0.2060319, 0.880797, 0.04742587, 0.11920291],
                [0.26714143, 0.00179998, 0.7261657, 0.98201376, 0.04742587, 0.95257413],
            ],
            [[0.21194156, 0.57611686, 0.99966466, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
        assert_almost_equal(entity_probability.detach().cpu().numpy(), true_probability)

    def get_fake_worlds(self):
        # Generate a toy WikitablesWorld.
        FakeTable = namedtuple("FakeTable", ["entities", "neighbors"])
        FakeWorld = namedtuple("FakeWorld", ["table_graph"])
        entities = [
            ["-1", "2010", "2012", "string:bmw", "date_column:year", "string_column:make"],
            ["-1", "2012", "date_column:year"],
        ]
        neighbors = [
            {
                "2010": ["date_column:year"],
                "2012": ["date_column:year"],
                "string:bmw": ["string_column:make"],
                "date_column:year": ["2010", "2012"],
                "string_column:make": ["string:bmw"],
                "-1": [],
            },
            {"2012": ["date_column:year"], "date_column:year": ["2012"], "-1": []},
        ]

        worlds = [
            FakeWorld(FakeTable(entity_list, entity2neighbors))
            for entity_list, entity2neighbors in zip(entities, neighbors)
        ]
        num_entities = max([len(entity_list) for entity_list in entities])
        return worlds, num_entities
