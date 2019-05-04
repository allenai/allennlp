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

