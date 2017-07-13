# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, TagField
from allennlp.data.token_indexers import token_indexers
from allennlp.common.checks import ConfigurationError
from allennlp.testing.test_case import AllenNlpTestCase


class TestTagField(AllenNlpTestCase):

    def test_tag_length_mismatch_raises(self):
        with pytest.raises(ConfigurationError):
            text = TextField(["here", "are", "some", "words", "."], [])
            wrong_tags = ["B", "O", "O"]
            _ = TagField(wrong_tags, text)

    def test_count_vocab_items_correctly_indexes_tags(self):
        text = TextField(["here", "are", "some", "words", "."], [token_indexers["single id"]("words")])
        tags = ["B", "I", "O", "O", "O"]
        tag_field = TagField(tags, text, tag_namespace="tags")

        counter = defaultdict(lambda: defaultdict(int))
        tag_field.count_vocab_items(counter)

        assert counter["tags"]["B"] == 1
        assert counter["tags"]["I"] == 1
        assert counter["tags"]["O"] == 3
        assert set(counter.keys()) == {"tags"}

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        b_index = vocab.add_token_to_namespace("B", namespace='*tags')
        i_index = vocab.add_token_to_namespace("I", namespace='*tags')
        o_index = vocab.add_token_to_namespace("O", namespace='*tags')

        text = TextField(["here", "are", "some", "words", "."], [token_indexers["single id"]("words")])
        tags = ["B", "I", "O", "O", "O"]
        tag_field = TagField(tags, text, tag_namespace="*tags")
        tag_field.index(vocab)

        # pylint: disable=protected-access
        assert tag_field._indexed_tags == [b_index, i_index, o_index, o_index, o_index]
        assert tag_field._num_tags == 3
        # pylint: enable=protected-access

    def test_pad_produces_one_hot_targets(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("B", namespace='*tags')
        vocab.add_token_to_namespace("I", namespace='*tags')
        vocab.add_token_to_namespace("O", namespace='*tags')

        text = TextField(["here", "are", "some", "words", "."], [token_indexers["single id"]("words")])
        tags = ["B", "I", "O", "O", "O"]
        tag_field = TagField(tags, text, tag_namespace="*tags")
        tag_field.index(vocab)
        padding_lengths = tag_field.get_padding_lengths()
        array = tag_field.as_array(padding_lengths)
        numpy.testing.assert_array_almost_equal(array, numpy.array([[1, 0, 0],
                                                                    [0, 1, 0],
                                                                    [0, 0, 1],
                                                                    [0, 0, 1],
                                                                    [0, 0, 1]]))
