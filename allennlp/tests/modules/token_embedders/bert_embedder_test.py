# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.bert_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask

class TestBertEmbedder(ModelTestCase):
    def test_without_offsets(self):
        embedder = PretrainedBertEmbedder('bert-base-uncased')
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids)

        assert list(result.shape) == [2, 5, 768]

    def test_with_offsets(self):
        embedder = PretrainedBertEmbedder('bert-base-uncased')
        input_ids = torch.LongTensor([[31, 51, 99, 17, 29], [15, 5, 0, 0, 0]])
        input_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 2, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = embedder(input_ids, input_mask, token_type_ids, offsets)

        assert list(result.shape) == [2, 3, 768]

    def test_end_to_end(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
        token_indexer = PretrainedBertIndexer('bert-base-uncased')

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
        tokens = tensor_dict["tokens"]


        mask = get_text_field_mask(tokens)
        token_type_ids = torch.zeros_like(mask)

        embedder = PretrainedBertEmbedder('bert-base-uncased')

        # No offsets, should get 11 vectors back.
        bert_vectors = embedder(tokens["bert"], mask, token_type_ids)
        assert list(bert_vectors.shape) == [2, 11, 768]

        # Offsets, should get 10 vectors back.
        bert_vectors = embedder(tokens["bert"], mask, token_type_ids, offsets=tokens["bert-offsets"])
        assert list(bert_vectors.shape) == [2, 10, 768]

