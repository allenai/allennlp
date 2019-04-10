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


class EmptyListTest(AllenNlpTestCase):
    def test_empty_list_can_be_tensorized(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize("Foo")
        text_field = TextField(tokens, token_indexers)
        list_field = ListField([text_field.empty_field()])
        fields = {'list': list_field}
        instance = Instance(fields)
        instance.as_tensor_dict()
