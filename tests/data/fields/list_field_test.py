# pylint: disable=no-self-use,invalid-name
import numpy

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.common.testing import AllenNlpTestCase


class TestListField(AllenNlpTestCase):

    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("this", "words")
        self.vocab.add_token_to_namespace("is", "words")
        self.vocab.add_token_to_namespace("a", "words")
        self.vocab.add_token_to_namespace("sentence", 'words')
        self.vocab.add_token_to_namespace("s", 'characters')
        self.vocab.add_token_to_namespace("e", 'characters')
        self.vocab.add_token_to_namespace("n", 'characters')
        self.vocab.add_token_to_namespace("t", 'characters')
        self.vocab.add_token_to_namespace("c", 'characters')

        self.word_indexer = {"words": SingleIdTokenIndexer("words")}
        self.words_and_characters_indexer = {"words": SingleIdTokenIndexer("words"),
                                             "characters": TokenCharactersIndexer("characters")}
        super(TestListField, self).setUp()

    def test_get_padding_lengths(self):
        field1 = TextField(["this", "is", "a", "sentence"], self.word_indexer)
        field2 = TextField(["this", "is", "a", "different", "sentence"], self.word_indexer)
        field3 = TextField(["this", "is", "another", "sentence"], self.word_indexer)

        list_field = ListField([field1, field2, field3])
        list_field.index(self.vocab)
        lengths = list_field.get_padding_lengths()

        assert lengths == {"num_fields": 3, "num_tokens": 5}

    def test_all_fields_padded_to_max_length(self):
        field1 = TextField(["this", "is", "a", "sentence"], self.word_indexer)
        field2 = TextField(["this", "is", "a", "different", "sentence"], self.word_indexer)
        field3 = TextField(["this", "is", "another", "sentence"], self.word_indexer)

        list_field = ListField([field1, field2, field3])
        list_field.index(self.vocab)

        array_dict = list_field.as_array(list_field.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(array_dict["words"][0], numpy.array([2, 3, 4, 5, 0]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][1], numpy.array([2, 3, 4, 1, 5]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][2], numpy.array([2, 3, 1, 5, 0]))

    def test_fields_can_pad_to_greater_than_max_length(self):

        field1 = TextField(["this", "is", "a", "sentence"], self.word_indexer)
        field2 = TextField(["this", "is", "a", "different", "sentence"], self.word_indexer)
        field3 = TextField(["this", "is", "another", "sentence"], self.word_indexer)

        list_field = ListField([field1, field2, field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        padding_lengths["num_tokens"] = 7
        padding_lengths["num_fields"] = 5
        array_dict = list_field.as_array(padding_lengths)
        numpy.testing.assert_array_almost_equal(array_dict["words"][0], numpy.array([2, 3, 4, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][1], numpy.array([2, 3, 4, 1, 5, 0, 0]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][2], numpy.array([2, 3, 1, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][3], numpy.array([0, 0, 0, 0, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(array_dict["words"][4], numpy.array([0, 0, 0, 0, 0, 0, 0]))

    def test_as_array_can_handle_multiple_token_indexers(self):
        field1 = TextField(["this", "is", "a", "sentence"], self.words_and_characters_indexer)
        field2 = TextField(["this", "is", "a", "different", "sentence"], self.words_and_characters_indexer)
        field3 = TextField(["this", "is", "another", "sentence"], self.words_and_characters_indexer)

        list_field = ListField([field1, field2, field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        array_dict = list_field.as_array(padding_lengths)
        words = array_dict["words"]
        characters = array_dict["characters"]
        numpy.testing.assert_array_almost_equal(words, numpy.array([[2, 3, 4, 5, 0],
                                                                    [2, 3, 4, 1, 5],
                                                                    [2, 3, 1, 5, 0]]))

        numpy.testing.assert_array_almost_equal(characters[0], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        numpy.testing.assert_array_almost_equal(characters[1], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 1, 1, 1, 3, 1, 3, 4, 5],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0]]))

        numpy.testing.assert_array_almost_equal(characters[2], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 4, 1, 5, 1, 3, 1, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))
