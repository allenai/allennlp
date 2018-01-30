# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.fields import KnowledgeGraphField
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter


class KnowledgeGraphFieldTest(AllenNlpTestCase):
    def setUp(self):
        json = {
                'columns': ['Name in English', 'Location in English'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        self.graph = TableKnowledgeGraph.read_from_json(json)
        self.vocab = Vocabulary()
        self.name_index = self.vocab.add_token_to_namespace("Name", namespace='tokens')
        self.in_index = self.vocab.add_token_to_namespace("in", namespace='tokens')
        self.english_index = self.vocab.add_token_to_namespace("English", namespace='tokens')
        self.location_index = self.vocab.add_token_to_namespace("Location", namespace='tokens')
        self.paradeniz_index = self.vocab.add_token_to_namespace("Paradeniz", namespace='tokens')
        self.mersin_index = self.vocab.add_token_to_namespace("Mersin", namespace='tokens')
        self.lake_index = self.vocab.add_token_to_namespace("Lake", namespace='tokens')
        self.gala_index = self.vocab.add_token_to_namespace("Gala", namespace='tokens')

        self.oov_index = self.vocab.get_token_index('random OOV string', namespace='tokens')
        self.edirne_index = self.oov_index

        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self.utterance = self.tokenizer.tokenize("Where is Mersin?")
        self.token_indexers = {"tokens": SingleIdTokenIndexer("tokens")}
        self.field = KnowledgeGraphField(self.graph, self.utterance, self.token_indexers, self.tokenizer)

        super(KnowledgeGraphFieldTest, self).setUp()

    def test_count_vocab_items(self):
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        self.field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["tokens"] == {
                'Name': 1,
                'in': 2,
                'English': 2,
                'Location': 1,
                'Paradeniz': 1,
                'Mersin': 1,
                'Lake': 1,
                'Gala': 1,
                'Edirne': 1,
                }

    def test_index_converts_field_correctly(self):
        # pylint: disable=protected-access
        self.field.index(self.vocab)
        assert self.field._indexed_entity_texts.keys() == {'tokens'}
        # Note that these are sorted by their _identifiers_, not their cell text, so the
        # `fb:row.rows` show up after the `fb:cells`.
        expected_array = [[self.edirne_index],
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
        assert self.field.get_padding_lengths() == {'num_entities': 6, 'num_entity_tokens': 3,
                                                    'num_utterance_tokens': 4}
        self.field._token_indexers['token_characters'] = TokenCharactersIndexer()
        self.field.index(self.vocab)
        assert self.field.get_padding_lengths() == {'num_entities': 6, 'num_entity_tokens': 3,
                                                    'num_utterance_tokens': 4,
                                                    'num_token_characters': 9}

    def test_as_tensor_produces_correct_output(self):
        self.field.index(self.vocab)
        padding_lengths = self.field.get_padding_lengths()
        padding_lengths['num_utterance_tokens'] += 1
        padding_lengths['num_entities'] += 1
        tensor_dict = self.field.as_tensor(padding_lengths)
        assert tensor_dict.keys() == {'text', 'linking'}
        expected_text_tensor = [[self.edirne_index, 0, 0],
                                [self.lake_index, self.gala_index, 0],
                                [self.mersin_index, 0, 0],
                                [self.paradeniz_index, 0, 0],
                                [self.location_index, self.in_index, self.english_index],
                                [self.name_index, self.in_index, self.english_index],
                                [0, 0, 0]]
        assert_almost_equal(tensor_dict['text']['tokens'].data.cpu().numpy(), expected_text_tensor)

        expected_linking_tensor = [[[0, 0, 0, 0, .2, 0, 0],  # fb:cell.edirne, "Where"
                                    [0, 0, 0, 0, -1.5, 0, 0],  # fb:cell.edirne, "is"
                                    [0, 0, 0, 0, 0, 0, 0],  # fb:cell.edirne, "Mersin"
                                    [0, 0, 0, 0, -5, 0, 0],  # fb:cell.edirne, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:cell.edirne, padding
                                   [[0, 0, 0, 0, -.6, 0, 0],  # fb:cell.lake_gala, "Where"
                                    [0, 0, 0, 0, -3.5, 0, 0],  # fb:cell.lake_gala, "is"
                                    [0, 0, 0, 0, -.3333333, 0, 0],  # fb:cell.lake_gala, "Mersin"
                                    [0, 0, 0, 0, -8, 0, 0],  # fb:cell.lake_gala, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:cell.lake_gala, padding
                                   [[0, 0, 0, 0, 0, 0, 0],  # fb:cell.mersin, "Where"
                                    [0, 0, 0, 0, -1.5, 0, 0],  # fb:cell.mersin, "is"
                                    [1, 1, 1, 1, 1, 0, 0],  # fb:cell.mersin, "Mersin"
                                    [0, 0, 0, 0, -5, 0, 0],  # fb:cell.mersin, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:cell.mersin, padding
                                   [[0, 0, 0, 0, -.6, 0, 0],  # fb:cell.paradeniz, "Where"
                                    [0, 0, 0, 0, -3, 0, 0],  # fb:cell.paradeniz, "is"
                                    [0, 0, 0, 0, -.1666666, 0, 0],  # fb:cell.paradeniz, "Mersin"
                                    [0, 0, 0, 0, -8, 0, 0],  # fb:cell.paradeniz, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:cell.paradeniz, padding
                                   [[0, 0, 0, 0, -2.8, 0, 0],  # fb:row.row.name_in_english, "Where"
                                    [0, 0, 0, 0, -7.5, 0, 0],  # fb:row.row.name_in_english, "is"
                                    [0, 0, 0, 0, -1.8333333, 1, 1],  # fb:row.row.name_in_english, "Mersin"
                                    [0, 0, 0, 0, -18, 0, 0],  # fb:row.row.name_in_english, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:row.row.name_in_english, padding
                                   [[0, 0, 0, 0, -1.8, 0, 0],  # fb:row.row.location_in_english, "Where"
                                    [0, 0, 0, 0, -5.5, 0, 0],  # fb:row.row.location_in_english, "is"
                                    [0, 0, 0, 0, -1.1666666, 0, 0],  # fb:row.row.location_in_english, "Mersin"
                                    [0, 0, 0, 0, -14, 0, 0],  # fb:row.row.location_in_english, "?"
                                    [0, 0, 0, 0, 0, 0, 0]],  # fb:row.row.location_in_english, padding
                                   [[0, 0, 0, 0, 0, 0, 0],  # padding, "Where"
                                    [0, 0, 0, 0, 0, 0, 0],  # padding, "is"
                                    [0, 0, 0, 0, 0, 0, 0],  # padding, "Mersin"
                                    [0, 0, 0, 0, 0, 0, 0],  # padding, "?"
                                    [0, 0, 0, 0, 0, 0, 0]]]  # padding, padding
        assert_almost_equal(tensor_dict['linking'].data.cpu().numpy(), expected_linking_tensor)

    def test_lemma_feature_extractor(self):
        # pylint: disable=protected-access
        utterance = self.tokenizer.tokenize("Names in English")
        field = KnowledgeGraphField(self.graph, self.utterance, self.token_indexers, self.tokenizer)
        entity = 'fb:row.row.name_in_english'
        assert field._contains_lemma_match(entity, field._entity_text_map[entity], utterance[0]) == 1

    def test_batch_tensors(self):
        self.field.index(self.vocab)
        padding_lengths = self.field.get_padding_lengths()
        tensor_dict1 = self.field.as_tensor(padding_lengths)
        tensor_dict2 = self.field.as_tensor(padding_lengths)
        batched_tensor_dict = self.field.batch_tensors([tensor_dict1, tensor_dict2])
        assert batched_tensor_dict.keys() == {'text', 'linking'}
        expected_single_tensor = [[self.edirne_index, 0, 0],
                                  [self.lake_index, self.gala_index, 0],
                                  [self.mersin_index, 0, 0],
                                  [self.paradeniz_index, 0, 0],
                                  [self.location_index, self.in_index, self.english_index],
                                  [self.name_index, self.in_index, self.english_index]]
        expected_batched_tensor = [expected_single_tensor, expected_single_tensor]
        assert_almost_equal(batched_tensor_dict['text']['tokens'].data.cpu().numpy(),
                            expected_batched_tensor)
        expected_linking_tensor = torch.stack([tensor_dict1['linking'], tensor_dict2['linking']])
        assert_almost_equal(batched_tensor_dict['linking'].data.cpu().numpy(),
                            expected_linking_tensor.data.cpu().numpy())
