# pylint: disable=no-self-use,invalid-name
import json
from typing import Iterator, List, Dict, Any, MutableMapping

import torch
from torch.nn import Module

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, DatasetReader, Field
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models import Model
from allennlp.modules import Embedding
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

    def text_to_instance(self, # type: ignore
                         body: str,
                         entity_name: str,
                         entity_mentions: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        """
        Generates an instance based on a body of text, an entity with a
        given name (which need not be in the body) and series of entity
        mentions. The mentions will be matched against the text. In the real
        model we generate spans, but for this repro we return them as
        TextFields.
        """

        fields: MutableMapping[str, Field[Any]] = {}

        body_tokens = self._tokenizer.tokenize(body)
        fields['body'] = TextField(body_tokens, self._token_indexers)

        EMPTY_TEXT = fields['body'].empty_field()

        def get_matching_entities(mentions):
            matched_mention = []
            for mention in mentions:
                mention_tokens = self._tokenizer.tokenize(mention)
                for start_index in range(0, len(body_tokens) - len(mention_tokens) + 1):
                    selected_tokens = body_tokens[start_index:start_index + len(mention_tokens)]
                    if self._is_same_token_sequence(selected_tokens, mention_tokens):
                        matched_mention.append(TextField(selected_tokens, self._token_indexers))
            # Empty lists fields are actually non-empty list fields full of padding.
            if not matched_mention:
                matched_mention.append(EMPTY_TEXT)
            return ListField(matched_mention)

        fields['entity_name'] = TextField(self._tokenizer.tokenize(entity_name), self._token_indexers)
        fields['entity_mentions'] = get_matching_entities(entity_mentions)

        return Instance(fields)

class FixedLengthEmbedding(Module):
    def forward(self, mask, embedded_tokens):
        # pylint: disable=arguments-differ
        """
        Create a very simple fixed length embedding of a sequence by
        concatenating the first and last embedded tokens.
        """
        sequence_lengths = mask.sum(dim=1)
        # size: <batch_size, emb_dim>
        embedded_first_tokens = embedded_tokens[:, 0, :]
        # size: <batch_size>
        indices = sequence_lengths - 1
        # size: <batch_size>
        zeros = torch.zeros_like(indices)
        # Handle empty lists. Caller responsible for masking.
        # size: <batch_size>
        adjusted_indices = torch.stack((indices, zeros), dim=1).max(dim=1)[0]
        # size: <batch_size, emb_dim>
        embedded_last_tokens = batched_index_select(embedded_tokens, adjusted_indices)
        # size: <batch_size, 2 * emb_dim>
        return torch.cat((embedded_first_tokens, embedded_last_tokens), 1)

class SalienceModel(Model):
    """
    An unsupervised baseline model for salience based on textual similarity.
    """
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        # Dummy weights
        weight = torch.ones(vocab.get_vocab_size(), 10)
        token_embedding = Embedding(
                num_embeddings=vocab.get_vocab_size(),
                embedding_dim=10,
                weight=weight,
                trainable=False)
        self.embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
        self.fixed_length_embedding = FixedLengthEmbedding()

    def forward(self, # type: ignore
                body: Dict[str, torch.LongTensor],
                entity_name: Dict[str, torch.LongTensor],
                entity_mentions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        # Embed body

        # size: <batch_size, sequence_len>
        body_mask = get_text_field_mask(body)
        # size: <batch_size, sequence_len, emb_dim>
        embedded_body_tokens = self.embedder(body)
        # size: <batch_size, 2 * emb_dim>
        embedded_body = self.fixed_length_embedding(body_mask, embedded_body_tokens)

        # Embed name (in isolation)

        # size: <batch_size, sequence_len>
        name_mask = get_text_field_mask(entity_name)
        # size: <batch_size, sequence_len, emb_dim>
        embedded_name_tokens = self.embedder(entity_name)
        # size: <batch_size, 2 * emb_dim>
        embedded_name = self.fixed_length_embedding(name_mask, embedded_name_tokens)

        # Extract embedded spans from the body

        # size: <batch_size, mentions_count, sequence_len>
        mentions_mask = get_text_field_mask(entity_mentions, num_wrapping_dims=1)
        # size: <batch_size, mentions_count, sequence_len, emb_dim>
        embedded_mentions_tokens = self.embedder(entity_mentions)
        # size: [<batch_size, 1, 2 * emb_dim>]
        embedded_spans_list = []
        for i in range(mentions_mask.size(1)):
            embedded_spans_tmp = self.fixed_length_embedding(
                    mentions_mask[:, i, :],
                    embedded_mentions_tokens[:, i, :, :]
            ).unsqueeze(1)
            embedded_spans_list.append(embedded_spans_tmp)
        # size: <batch_size, mentions_count, 2 * emb_dim>
        embedded_spans = torch.cat(embedded_spans_list, 1)

        # size: <batch_size>
        name_match_score = torch.nn.functional.cosine_similarity(embedded_body, embedded_name)

        # size: <batch_size, 2 * emb_dim, mentions_count>
        transposed_embedded_spans = embedded_spans.transpose(1, 2)
        # Note: Real model normalizes to give cosine similarity.
        # size: <batch_size, mentions_count>
        span_match_scores = torch.bmm(embedded_body.unsqueeze(1), transposed_embedded_spans).squeeze(1)
        # size: <batch_size, mentions_count>
        masked_span_match_scores = span_match_scores * (mentions_mask[:, :, 0] != 0).float()
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
        fields = {'list': list_field, 'bar': TextField(tokenizer.tokenize("BAR"), token_indexers)}
        instance = Instance(fields)
        vocab = Vocabulary.from_instances([instance])
        instance.index_fields(vocab)
        instance.as_tensor_dict()

    # A batch with entirely empty lists.
    def test_end_to_end_broken_without_fix(self):
        reader = SalienceReader()
        dataset = reader.read(self.FIXTURES_ROOT / 'end_to_end' / 'sample.json')[1:]
        vocab = Vocabulary.from_instances(dataset)
        model = SalienceModel(vocab)
        model.eval()
        iterator = BasicIterator(batch_size=2)
        iterator.index_with(vocab)
        batch = next(iterator(dataset, shuffle=False))
        model.forward(**batch)

    # A mixed batch with some empty lists.
    def test_end_to_end_works_in_master(self):
        reader = SalienceReader()
        dataset = reader.read(self.FIXTURES_ROOT / 'end_to_end' / 'sample.json')
        vocab = Vocabulary.from_instances(dataset)
        model = SalienceModel(vocab)
        model.eval()
        iterator = BasicIterator(batch_size=2)
        iterator.index_with(vocab)
        batch = next(iterator(dataset, shuffle=False))
        results = model.forward(**batch)["score"]
        # For the sample data:
        # {"body": "This is a test.", "entity_name": "exam", "entity_mentions": ["test", "quiz"]}
        # {"body": "The dog went on a walk.", "entity_name": "animal", "entity_mentions": ["hound", "puppy"]}
        assert results[0] > results[1]
