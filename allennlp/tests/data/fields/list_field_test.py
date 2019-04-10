# pylint: disable=no-self-use,invalid-name
import json
import torch
from typing import Iterator, List, Dict

import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, DatasetReader
from allennlp.data.fields import TextField, LabelField, ListField, IndexField, SequenceLabelField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, batched_index_select


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
                                              "characters": TokenCharactersIndexer("characters",
                                                                                   min_padding_length=1)}
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
        assert lengths == {"num_fields": 3, "list_words_length": 5, "list_num_tokens": 5}

    def test_list_field_can_handle_empty_text_fields(self):
        list_field = ListField([self.field1, self.field2, self.empty_text_field])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor_dict["words"].detach().cpu().numpy(),
                                         numpy.array([[2, 3, 4, 5, 0],
                                                      [2, 3, 4, 1, 5],
                                                      [0, 0, 0, 0, 0]]))

    def test_list_field_can_handle_empty_index_fields(self):
        list_field = ListField([self.index_field, self.index_field, self.empty_index_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.detach().cpu().numpy(), numpy.array([[1], [1], [-1]]))

    def test_list_field_can_handle_empty_sequence_label_fields(self):
        list_field = ListField([self.sequence_label_field,
                                self.sequence_label_field,
                                self.empty_sequence_label_field])
        list_field.index(self.vocab)
        tensor = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_equal(tensor.detach().cpu().numpy(),
                                         numpy.array([[1, 1, 0, 1],
                                                      [1, 1, 0, 1],
                                                      [0, 0, 0, 0]]))

    def test_all_fields_padded_to_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][0].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][1].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][2].detach().cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0]))

    def test_nested_list_fields_are_padded_correctly(self):
        nested_field1 = ListField([LabelField(c) for c in ['a', 'b', 'c', 'd', 'e']])
        nested_field2 = ListField([LabelField(c) for c in ['f', 'g', 'h', 'i', 'j', 'k']])
        list_field = ListField([nested_field1.empty_field(), nested_field1, nested_field2])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        assert padding_lengths == {'num_fields': 3, 'list_num_fields': 6}
        tensor = list_field.as_tensor(padding_lengths).detach().cpu().numpy()
        numpy.testing.assert_almost_equal(tensor, [[-1, -1, -1, -1, -1, -1],
                                                   [0, 1, 2, 3, 4, -1],
                                                   [5, 6, 7, 8, 9, 10]])

    def test_fields_can_pad_to_greater_than_max_length(self):
        list_field = ListField([self.field1, self.field2, self.field3])
        list_field.index(self.vocab)
        padding_lengths = list_field.get_padding_lengths()
        padding_lengths["list_words_length"] = 7
        padding_lengths["num_fields"] = 5
        tensor_dict = list_field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][0].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][1].detach().cpu().numpy(),
                                                numpy.array([2, 3, 4, 1, 5, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][2].detach().cpu().numpy(),
                                                numpy.array([2, 3, 1, 5, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][3].detach().cpu().numpy(),
                                                numpy.array([0, 0, 0, 0, 0, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["words"][4].detach().cpu().numpy(),
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
        words = tensor_dict["words"].detach().cpu().numpy()
        characters = tensor_dict["characters"].detach().cpu().numpy()
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
        words = tensor_dict["words"].detach().cpu().numpy()
        characters = tensor_dict["characters"].detach().cpu().numpy()

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

    def test_printing_doesnt_crash(self):
        list_field = ListField([self.field1, self.field2])
        print(list_field)

    def test_sequence_methods(self):
        list_field = ListField([self.field1, self.field2, self.field3])

        assert len(list_field) == 3
        assert list_field[1] == self.field2
        assert [f for f in list_field] == [self.field1, self.field2, self.field3]

    def test_empty_list_can_be_tensorized(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize("Foo")
        text_field = TextField(tokens, token_indexers)
        list_field = ListField([text_field.empty_field()])
        fields = {'list': list_field}
        instance = Instance(fields)
        instance.as_tensor_dict()

    class SalienceReader(DatasetReader):
        def __init__(self):
            super().__init__(lazy=False)
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
            self._tokenizer = WordTokenizer()

        def _read(self, file_path: str) -> Iterator[Instance]:
            with open(file_path) as file:
                for line in file:
                    doc = json.loads(line)
                    yield self.text_to_instance(body=doc['body'],
                                                entity_name=doc['entity_name'],
                                                entity_mentions=doc['entity_mentions'])

        @classmethod
        def _is_same_token_sequence(cls, seq1: List[Token], seq2: List[Token]):
            """
            Utility function to check if two token sequences are identical.
            """
            for t1, t2 in zip(seq1, seq2):
                if t1.text != t2.text:
                    return False
            return True

        def text_to_instance(self,
                             body: str,
                             entity_name: str,
                             entity_mentions: List[str]) -> Instance:
            """
            Generates an instance based on a body of text, an entity with a
            given name (which need not be in the body) and series of entity
            mentions. The mentions will be matched against the text to generate
            mention spans.
            """

            fields = {}

            body_tokens = self._tokenizer.tokenize(body)
            fields['body'] = TextField(body_tokens, self._token_indexers)

            EMPTY_TEXT = fields['body'].empty_field()
            EMPTY_SPAN = SpanField(-1, -1, EMPTY_TEXT)

            def get_entity_spans(mentions):
                spans = []
                for mention in mentions:
                    mention_tokens = self._tokenizer.tokenize(mention)
                    for start_index in range(0, len(body_tokens) - len(mention_tokens) + 1):
                        selected_tokens = body_tokens[start_index:start_index + len(mention_tokens)]
                        if self._is_same_token_sequence(selected_tokens, mention_tokens):
                            spans.append(SpanField(start_index,
                                                   start_index + len(mention_tokens) - 1,
                                                   fields['body']))
                # Empty lists fields are actually non-empty list fields full of padding.
                if not spans:
                    spans.append(EMPTY_SPAN)
                return ListField(spans)

            fields['entity_name'] = TextField(self._tokenizer.tokenize(entity_name), self._token_indexers)
            fields['entity_spans'] = get_entity_spans(entity_mentions)

            return Instance(fields)

    class SalienceModel(Model):
        """
        An unsupervised baseline model for salience based on textual similarity.
        """
        def __init__(self, vocab: Vocabulary) -> None:
            super().__init__(vocab)
            # In the real model this is pretrained.
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=30)
            self._embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        @classmethod
        def fixed_length_embedding(cls, mask, embedded_tokens):
            """
            Create a very simple fixed length embedding of a sequence by
            concatenating the first and last embedded tokens.
            """
            sequence_lengths = mask.sum(dim=1)
            # size: <batch_size, emb_dim>
            embedded_first_tokens = embedded_tokens[:,0,:]
            # size: <batch_size>
            indices = sequence_lengths - 1
            # size: <batch_size, emb_dim>
            embedded_last_tokens = batched_index_select(embedded_tokens, indices)
            # size: <batch_size, 2 * emb_dim>
            return torch.cat((embedded_first_tokens, embedded_last_tokens), 1)

        def forward(self,
                    body: Dict[str, torch.LongTensor],
                    entity_name: Dict[str, torch.LongTensor],
                    entity_spans: torch.Tensor) -> Dict[str, torch.Tensor]:

            # Embed body

            # size: <batch_size, sequence_len>
            body_mask = get_text_field_mask(body)
            # size: <batch_size, sequence_len, emb_dim>
            embedded_body_tokens = self._embedder(body)
            # size: <batch_size, 2 * emb_dim>
            embedded_body = self.fixed_length_embedding(body_mask, embedded_body_tokens)

            # Embed name (in isolation)

            # size: <batch_size, sequence_len>
            name_mask = get_text_field_mask(entity_name)
            # size: <batch_size, sequence_len, emb_dim>
            embedded_name_tokens = self._embedder(entity_name)
            # size: <batch_size, 2 * emb_dim>
            embedded_name = self.fixed_length_embedding(name_mask, embedded_name_tokens)

            # Extract embedded spans from the body

            extractor = EndpointSpanExtractor(input_dim=embedded_body_tokens.size(-1))
            # size: <batch_size, mentions_count>
            span_mask = (entity_spans[:, :, 0] >= 0).long()
            # size: <batch_size, mentions_count, 2 * emb_dim>
            embedded_spans = extractor(embedded_body_tokens, entity_spans, span_indices_mask=span_mask)

            # size: <batch_size>
            name_match_score = torch.cosine_similarity(embedded_body, embedded_name)

            # size: <batch_size, 2 * emb_dim, mentions_count>
            transposed_embedded_spans = embedded_spans.transpose(1, 2)
            # Note: Real model normalizes to give cosine similarity.
            # size: <batch_size, mentions_count>
            span_match_scores = torch.matmul(embedded_body, transposed_embedded_spans)
            # size: <batch_size, mentions_count>
            masked_span_match_scores = span_match_scores * span_mask
            # Aggregate with max to get single score
            # size: <batch_size>
            span_match_score = masked_span_match_scores.max(dim=-1)[0].squeeze(-1)

            # Combine name match and span match scores.
            return {'score': name_match_score + span_match_score}
