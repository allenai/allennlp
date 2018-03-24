# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSequenceLabelField(AllenNlpTestCase):
    def setUp(self):
        super(TestSequenceLabelField, self).setUp()
        self.text = TextField([Token(t) for t in ["here", "are", "some", "words", "."]],
                              {"words": SingleIdTokenIndexer("words")})

    def test_tag_length_mismatch_raises(self):
        with pytest.raises(ConfigurationError):
            wrong_tags = ["B", "O", "O"]
            _ = SequenceLabelField(wrong_tags, self.text)

    def test_count_vocab_items_correctly_indexes_tags(self):
        tags = ["B", "I", "O", "O", "O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="labels")

        counter = defaultdict(lambda: defaultdict(int))
        sequence_label_field.count_vocab_items(counter)

        assert counter["labels"]["B"] == 1
        assert counter["labels"]["I"] == 1
        assert counter["labels"]["O"] == 3
        assert set(counter.keys()) == {"labels"}

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        b_index = vocab.add_token_to_namespace("B", namespace='*labels')
        i_index = vocab.add_token_to_namespace("I", namespace='*labels')
        o_index = vocab.add_token_to_namespace("O", namespace='*labels')

        tags = ["B", "I", "O", "O", "O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="*labels")
        sequence_label_field.index(vocab)

        # pylint: disable=protected-access
        assert sequence_label_field._indexed_labels == [b_index, i_index, o_index, o_index, o_index]
        # pylint: enable=protected-access

    def test_as_tensor_produces_integer_targets(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("B", namespace='*labels')
        vocab.add_token_to_namespace("I", namespace='*labels')
        vocab.add_token_to_namespace("O", namespace='*labels')

        tags = ["B", "I", "O", "O", "O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="*labels")
        sequence_label_field.index(vocab)
        padding_lengths = sequence_label_field.get_padding_lengths()
        tensor = sequence_label_field.as_tensor(padding_lengths).data.cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 1, 2, 2, 2]))

    def test_sequence_label_field_raises_on_incorrect_type(self):

        with pytest.raises(ConfigurationError):
            _ = SequenceLabelField([[], [], [], [], []], self.text)

    def test_class_variables_for_namespace_warnings_work_correctly(self):
        # pylint: disable=protected-access
        tags = ["B", "I", "O", "O", "O"]
        assert "text" not in SequenceLabelField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.sequence_label_field", level="WARNING"):
            _ = SequenceLabelField(tags, self.text, label_namespace="text")

        # We've warned once, so we should have set the class variable to False.
        assert "text" in SequenceLabelField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger="allennlp.data.fields.sequence_label_field", level="WARNING"):
                _ = SequenceLabelField(tags, self.text, label_namespace="text")

        # ... but a new namespace should still log a warning.
        assert "text2" not in SequenceLabelField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.sequence_label_field", level="WARNING"):
            _ = SequenceLabelField(tags, self.text, label_namespace="text2")

    def test_printing_doesnt_crash(self):
        tags = ["B", "I", "O", "O", "O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="labels")
        print(sequence_label_field)
