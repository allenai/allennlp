# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.bert_indexer import BertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertBaseUncased


class TestBertEmbedder(ModelTestCase):
    def test_with_random_weights(self):
        embedder = BertBaseUncased()
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids)

        assert list(result.shape) == [2, 5, 768]

    def test_with_offsets(self):
        embedder = BertBaseUncased()
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids, offsets)

        assert list(result.shape) == [2, 3, 768]

    def test_end_to_end(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        token_indexer = BertIndexer(vocab_path="/Users/joelg/data/uncased_L-12_H-768_A-12/vocab.txt")

        sentence1 = "The quick brown Bert jumped over the lazy ELMo."
        tokens1 = tokenizer.tokenize(sentence1)
        sentence2 = "Thanks to huggingface for implementing this."
        tokens2 = tokenizer.tokenize(sentence2)

        vocab = Vocabulary()

        instance1 = Instance({"tokens": TextField(tokens1, {"bert": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"bert": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
