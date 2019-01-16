# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.fields import KnowledgeGraphField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph


class KnowledgeGraphFieldTest(AllenNlpTestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self.utterance = self.tokenizer.tokenize("where is mersin?")
        self.token_indexers = {"tokens": SingleIdTokenIndexer("tokens")}

        json = {
                'question': self.utterance,
                'columns': ['Name in English', 'Location in English'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        self.graph = TableQuestionKnowledgeGraph.read_from_json(json)
        self.vocab = Vocabulary()
        self.name_index = self.vocab.add_token_to_namespace("name", namespace='tokens')
        self.in_index = self.vocab.add_token_to_namespace("in", namespace='tokens')
        self.english_index = self.vocab.add_token_to_namespace("english", namespace='tokens')
        self.location_index = self.vocab.add_token_to_namespace("location", namespace='tokens')
        self.paradeniz_index = self.vocab.add_token_to_namespace("paradeniz", namespace='tokens')
        self.mersin_index = self.vocab.add_token_to_namespace("mersin", namespace='tokens')
        self.lake_index = self.vocab.add_token_to_namespace("lake", namespace='tokens')
        self.gala_index = self.vocab.add_token_to_namespace("gala", namespace='tokens')
        self.negative_one_index = self.vocab.add_token_to_namespace("-1", namespace='tokens')
        self.zero_index = self.vocab.add_token_to_namespace("0", namespace='tokens')
        self.one_index = self.vocab.add_token_to_namespace("1", namespace='tokens')

        self.oov_index = self.vocab.get_token_index('random OOV string', namespace='tokens')
        self.edirne_index = self.oov_index
        self.field = KnowledgeGraphField(self.graph, self.utterance, self.token_indexers, self.tokenizer)

        super(KnowledgeGraphFieldTest, self).setUp()

    def test_count_vocab_items(self):
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        self.field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["tokens"] == {
                '-1': 1,
                '0': 1,
                '1': 1,
                'name': 1,
                'in': 2,
                'english': 2,
                'location': 1,
                'paradeniz': 1,
                'mersin': 1,
                'lake': 1,
                'gala': 1,
                'edirne': 1,
                }

    def test_index_converts_field_correctly(self):
        # pylint: disable=protected-access
        self.field.index(self.vocab)
        assert self.field._indexed_entity_texts.keys() == {'tokens'}
        # Note that these are sorted by their _identifiers_, not their cell text, so the
        # `fb:row.rows` show up after the `fb:cells`.
        expected_array = [[self.negative_one_index],
                          [self.zero_index],
                          [self.one_index],
                          [self.edirne_index],
                          [self.lake_index, self.gala_index],
                          [self.mersin_index],
                          [self.paradeniz_index],
                          [self.location_index, self.in_index, self.english_index],
                          [self.name_index, self.in_index, self.english_index]]
        assert self.field._indexed_entity_texts['tokens'] == expected_array

    def test_get_padding_lengths_raises_if_not_indexed(self):
        with pytest.raises(AssertionError):
            self.field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        # pylint: disable=protected-access
        self.field.index(self.vocab)
        assert self.field.get_padding_lengths() == {'num_entities': 9, 'num_entity_tokens': 3,
                                                    'num_utterance_tokens': 4}
        self.field._token_indexers['token_characters'] = TokenCharactersIndexer(min_padding_length=1)
        self.field.index(self.vocab)
        assert self.field.get_padding_lengths() == {'num_entities': 9, 'num_entity_tokens': 3,
                                                    'num_utterance_tokens': 4,
                                                    'num_token_characters': 9}

    def test_as_tensor_produces_correct_output(self):
        self.field.index(self.vocab)
        padding_lengths = self.field.get_padding_lengths()
        padding_lengths['num_utterance_tokens'] += 1
        padding_lengths['num_entities'] += 1
        tensor_dict = self.field.as_tensor(padding_lengths)
        assert tensor_dict.keys() == {'text', 'linking'}
        expected_text_tensor = [[self.negative_one_index, 0, 0],
                                [self.zero_index, 0, 0],
                                [self.one_index, 0, 0],
                                [self.edirne_index, 0, 0],
                                [self.lake_index, self.gala_index, 0],
                                [self.mersin_index, 0, 0],
                                [self.paradeniz_index, 0, 0],
                                [self.location_index, self.in_index, self.english_index],
                                [self.name_index, self.in_index, self.english_index],
                                [0, 0, 0]]
        assert_almost_equal(tensor_dict['text']['tokens'].detach().cpu().numpy(), expected_text_tensor)

        linking_tensor = tensor_dict['linking'].detach().cpu().numpy()
        expected_linking_tensor = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1, "where"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1, "is"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -1, "mersin"
                                    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0]],  # -1, "?"
                                   [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0, "where"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0, "is"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0, "mersin"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # 0, "?"
                                   [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1, "where"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1, "is"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1, "mersin"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # 1, "?"
                                   [[0, 0, 0, 0, 0, .2, 0, 0, 0, 0],  # fb:cell.edirne, "where"
                                    [0, 0, 0, 0, 0, -1.5, 0, 0, 0, 0],  # fb:cell.edirne, "is"
                                    [0, 0, 0, 0, 0, .1666, 0, 0, 0, 0],  # fb:cell.edirne, "mersin"
                                    [0, 0, 0, 0, 0, -5, 0, 0, 0, 0],  # fb:cell.edirne, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:cell.edirne, padding
                                   [[0, 0, 0, 0, 0, -.6, 0, 0, 0, 0],  # fb:cell.lake_gala, "where"
                                    [0, 0, 0, 0, 0, -3.5, 0, 0, 0, 0],  # fb:cell.lake_gala, "is"
                                    [0, 0, 0, 0, 0, -.3333, 0, 0, 0, 0],  # fb:cell.lake_gala, "mersin"
                                    [0, 0, 0, 0, 0, -8, 0, 0, 0, 0],  # fb:cell.lake_gala, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:cell.lake_gala, padding
                                   [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fb:cell.mersin, "where"
                                    [0, 0, 0, 0, 0, -1.5, 0, 0, 0, 0],  # fb:cell.mersin, "is"
                                    [0, 1, 1, 1, 1, 1, 0, 0, 1, 1],  # fb:cell.mersin, "mersin"
                                    [0, 0, 0, 0, 0, -5, 0, 0, 0, 0],  # fb:cell.mersin, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:cell.mersin, padding
                                   [[0, 0, 0, 0, 0, -.6, 0, 0, 0, 0],  # fb:cell.paradeniz, "where"
                                    [0, 0, 0, 0, 0, -3, 0, 0, 0, 0],  # fb:cell.paradeniz, "is"
                                    [0, 0, 0, 0, 0, -.1666, 0, 0, 0, 0],  # fb:cell.paradeniz, "mersin"
                                    [0, 0, 0, 0, 0, -8, 0, 0, 0, 0],  # fb:cell.paradeniz, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:cell.paradeniz, padding
                                   [[0, 0, 0, 0, 0, -2.6, 0, 0, 0, 0],  # fb:row.row.name_in_english, "where"
                                    [0, 0, 0, 0, 0, -7.5, 0, 0, 0, 0],  # fb:row.row.name_in_english, "is"
                                    [0, 0, 0, 0, 0, -1.8333, 1, 1, 0, 0],  # fb:row.row.name_in_english, "mersin"
                                    [0, 0, 0, 0, 0, -18, 0, 0, 0, 0],  # fb:row.row.name_in_english, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:row.row.name_in_english, padding
                                   [[0, 0, 0, 0, 0, -1.6, 0, 0, 0, 0],  # fb:row.row.location_in_english, "where"
                                    [0, 0, 0, 0, 0, -5.5, 0, 0, 0, 0],  # fb:row.row.location_in_english, "is"
                                    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # fb:row.row.location_in_english, "mersin"
                                    [0, 0, 0, 0, 0, -14, 0, 0, 0, 0],  # fb:row.row.location_in_english, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # fb:row.row.location_in_english, padding
                                   [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding, "where"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding, "is"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding, "mersin"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding, "?"
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]  # padding, padding
        for entity_index, entity_features in enumerate(expected_linking_tensor):
            for question_index, feature_vector in enumerate(entity_features):
                assert_almost_equal(linking_tensor[entity_index, question_index],
                                    feature_vector,
                                    decimal=4,
                                    err_msg=f"{entity_index} {question_index}")

    def test_lemma_feature_extractor(self):
        # pylint: disable=protected-access
        utterance = self.tokenizer.tokenize("Names in English")
        field = KnowledgeGraphField(self.graph, self.utterance, self.token_indexers, self.tokenizer)
        entity = 'fb:row.row.name_in_english'
        lemma_feature = field._contains_lemma_match(entity,
                                                    field._entity_text_map[entity],
                                                    utterance[0],
                                                    0,
                                                    utterance)
        assert lemma_feature == 1

    def test_span_overlap_fraction(self):
        # pylint: disable=protected-access
        utterance = self.tokenizer.tokenize("what is the name in english of mersin?")
        field = KnowledgeGraphField(self.graph, self.utterance, self.token_indexers, self.tokenizer)
        entity = 'fb:row.row.name_in_english'
        entity_text = field._entity_text_map[entity]
        feature_values = [field._span_overlap_fraction(entity, entity_text, token, i, utterance)
                          for i, token in enumerate(utterance)]
        assert feature_values == [0, 0, 0, 1, 1, 1, 0, 0, 0]

    def test_batch_tensors(self):
        self.field.index(self.vocab)
        padding_lengths = self.field.get_padding_lengths()
        tensor_dict1 = self.field.as_tensor(padding_lengths)
        tensor_dict2 = self.field.as_tensor(padding_lengths)
        batched_tensor_dict = self.field.batch_tensors([tensor_dict1, tensor_dict2])
        assert batched_tensor_dict.keys() == {'text', 'linking'}
        expected_single_tensor = [[self.negative_one_index, 0, 0],
                                  [self.zero_index, 0, 0],
                                  [self.one_index, 0, 0],
                                  [self.edirne_index, 0, 0],
                                  [self.lake_index, self.gala_index, 0],
                                  [self.mersin_index, 0, 0],
                                  [self.paradeniz_index, 0, 0],
                                  [self.location_index, self.in_index, self.english_index],
                                  [self.name_index, self.in_index, self.english_index]]
        expected_batched_tensor = [expected_single_tensor, expected_single_tensor]
        assert_almost_equal(batched_tensor_dict['text']['tokens'].detach().cpu().numpy(),
                            expected_batched_tensor)
        expected_linking_tensor = torch.stack([tensor_dict1['linking'], tensor_dict2['linking']])
        assert_almost_equal(batched_tensor_dict['linking'].detach().cpu().numpy(),
                            expected_linking_tensor.detach().cpu().numpy())
