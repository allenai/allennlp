# pylint: disable=no-self-use,invalid-name
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField, LabelField, ListField, IndexField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer


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
        for label in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
            self.vocab.add_token_to_namespace(label, 'labels')

        self.word_indexer = {"words": SingleIdTokenIndexer("words")}
        self.words_and_characters_indexers = {"words": SingleIdTokenIndexer("words"),
                                              "characters": TokenCharactersIndexer("characters")}
        self.field1 = TextField([Token(t) for t in ["this", "is", "a", "sentence"]],
                                self.word_indexer)
        self.field2 = TextField([Token(t) for t in ["this", "is", "a", "different", "sentence"]],
                                self.word_indexer)
        self.field3 = TextField([Token(t) for t in ["this", "is", "another", "sentence"]],
                                self.word_indexer)

        self.empty_text_field = self.field1.empty_field()
        self.index_field = IndexField(1, self.field1)
        self.empty_index_field = self.index_field.empty_field()
        self.sequence_label_field = SequenceLabelField([1, 1, 0, 1], self.field1)
        self.empty_sequence_label_field = self.sequence_label_field.empty_field()

        super(TestListField, self).setUp()

    def test_get_padding_lengths(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        lengths = list_field.get_padding_lengths()
        assert lengths == {"num_fields": 3, "list_num_tokens": 5}

    def test_list_field_can_handle_empty_text_fields(self):
        list_field = ListField([self.field1, self.field2, self.empty_text_field])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor_dict["words"].data.cpu().numpy(),
                                         numpy.array([[2, 3, 4, 5, 0],
                                                      [2, 3, 4, 1, 5],
                                                      [0, 0, 0, 0, 0]]))

    def test_list_field_can_handle_empty_index_fields(self):
        list_field = ListField([self.index_field, self.index_field, self.empty_index_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.data.cpu().numpy(), numpy.array([[1], [1], [-1]]))

    def test_list_field_can_handle_empty_sequence_label_fields(self):
        list_field = ListField([self.sequence_label_field,
                                self.sequence_label_field,
                                self.empty_sequence_label_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.data.cpu().numpy(),
                                         numpy.array([[1, 1, 0, 1],
                                                      [1, 1, 0, 1],
                                                      [0, 0, 0, 0]]))

    def test_all_fields_padded_to_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][0].data.cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][1].data.cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][2].data.cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0]))

    def test_nested_list_fields_are_padded_correctly(self):
        nested_field1 = ListField([LabelField(c) for c in ['a', 'b', 'c', 'd', 'e']])
        nested_field2 = ListField([LabelField(c) for c in ['f', 'g', 'h', 'i', 'j', 'k']])
        list_field = ListField([nested_field1.empty_field(), nested_field1, nested_field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        assert padding_lengths == {'num_fields': 3, 'list_num_fields': 6}
        tensor = list_field.as_tensor(padding_lengths).data.cpu().numpy()
        numpy.testing.assert_almost_equal(tensor, [[[-1], [-1], [-1], [-1], [-1], [-1]],
                                                   [[0], [1], [2], [3], [4], [-1]],
                                                   [[5], [6], [7], [8], [9], [10]]])

    def test_fields_can_pad_to_greater_than_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        padding_lengths["list_num_tokens"] = 7
        padding_lengths["num_fields"] = 5
        tensor_dict = list_field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][0].data.cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][1].data.cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][2].data.cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][3].data.cpu().numpy(),
                                                numpy.array([0, 0, 0, 0, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][4].data.cpu().numpy(),
                                                numpy.array([0, 0, 0, 0, 0, 0, 0]))

    def test_as_tensor_can_handle_multiple_token_indexers(self):
        # pylint: disable=protected-access
        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict["words"].data.cpu().numpy()
        characters = tensor_dict["characters"].data.cpu().numpy()
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

    def test_as_tensor_can_handle_multiple_token_indexers_and_empty_fields(self):
        # pylint: disable=protected-access
        self.field1._token_indexers = self.words_and_characters_indexers
        self.field2._token_indexers = self.words_and_characters_indexers
        self.field3._token_indexers = self.words_and_characters_indexers

        list_field = ListField([self.field1.empty_field(), self.field1, self.field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        tensor_dict = list_field.as_tensor(padding_lengths)
        words = tensor_dict["words"].data.cpu().numpy()
        characters = tensor_dict["characters"].data.cpu().numpy()

        numpy.testing.assert_array_almost_equal(words, numpy.array([[0, 0, 0, 0, 0],
                                                                    [2, 3, 4, 5, 0],
                                                                    [2, 3, 4, 1, 5]]))

        numpy.testing.assert_array_almost_equal(characters[0], numpy.zeros([5, 9]))

        numpy.testing.assert_array_almost_equal(characters[1], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0],
                                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        numpy.testing.assert_array_almost_equal(characters[2], numpy.array([[5, 1, 1, 2, 0, 0, 0, 0, 0],
                                                                            [1, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                            [1, 1, 1, 1, 3, 1, 3, 4, 5],
                                                                            [2, 3, 4, 5, 3, 4, 6, 3, 0]]))
