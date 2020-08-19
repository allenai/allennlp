import pytest
import numpy
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import AdjacencyField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Vocabulary, Token


class TestAdjacencyField(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.text = TextField(
            [Token(t) for t in ["here", "is", "a", "sentence", "."]],
            {"words": SingleIdTokenIndexer("words")},
        )

    def test_adjacency_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("a", namespace="labels")
        vocab.add_token_to_namespace("b", namespace="labels")
        vocab.add_token_to_namespace("c", namespace="labels")

        labels = ["a", "b"]
        indices = [(0, 1), (2, 1)]
        adjacency_field = AdjacencyField(indices, self.text, labels)
        adjacency_field.index(vocab)
        tensor = adjacency_field.as_tensor(adjacency_field.get_padding_lengths())
        numpy.testing.assert_equal(
            tensor.numpy(),
            numpy.array(
                [
                    [-1, 0, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1],
                ]
            ),
        )

    def test_adjacency_field_raises_with_out_of_bounds_indices(self):
        with pytest.raises(ConfigurationError):
            _ = AdjacencyField([(0, 24)], self.text)

    def test_adjacency_field_raises_with_mismatching_labels_for_indices(self):
        with pytest.raises(ConfigurationError):
            _ = AdjacencyField([(0, 1), (0, 2)], self.text, ["label1"])

    def test_adjacency_field_raises_with_duplicate_indices(self):
        with pytest.raises(ConfigurationError):
            _ = AdjacencyField([(0, 1), (0, 1)], self.text, ["label1"])

    def test_adjacency_field_empty_field_works(self):
        field = AdjacencyField([(0, 1)], self.text)
        empty_field = field.empty_field()
        assert empty_field.indices == []

    def test_printing_doesnt_crash(self):
        adjacency_field = AdjacencyField([(0, 1)], self.text, ["label1"])
        print(adjacency_field)
