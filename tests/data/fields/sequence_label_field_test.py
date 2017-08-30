# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase


class TestSequenceLabelField(AllenNlpTestCase):

    def setUp(self):
        super(TestSequenceLabelField, self).setUp()
        self.text = TextField(["here", "are", "some", "words", "."],
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

    def test_as_array_produces_integer_targets(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("B", namespace='*labels')
        vocab.add_token_to_namespace("I", namespace='*labels')
        vocab.add_token_to_namespace("O", namespace='*labels')

        tags = ["B", "I", "O", "O", "O"]
        sequence_label_field = SequenceLabelField(tags, self.text, label_namespace="*labels")
        sequence_label_field.index(vocab)
        padding_lengths = sequence_label_field.get_padding_lengths()
        array = sequence_label_field.as_array(padding_lengths)
        numpy.testing.assert_array_almost_equal(array, numpy.array([0, 1, 2, 2, 2]))

    def test_sequence_label_field_raises_on_incorrect_type(self):

        with pytest.raises(ConfigurationError):
            _ = SequenceLabelField([[], [], [], [], []], self.text)
