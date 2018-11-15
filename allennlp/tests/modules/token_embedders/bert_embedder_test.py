# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder


class TestBertEmbedder(ModelTestCase):
    def test_with_random_weights(self):
        embedder = BertEmbedder(vocab_size=50000, hidden_size=768)
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids)

        assert list(result.shape) == [2, 5, 768]

    def test_with_offsets(self):
        embedder = BertEmbedder(vocab_size=50000, hidden_size=768)
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids, offsets)

        assert list(result.shape) == [2, 3, 768]
