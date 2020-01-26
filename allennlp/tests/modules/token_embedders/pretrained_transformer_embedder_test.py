import pytest
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestPretrainedTransformerEmbedder(AllenNlpTestCase):
    def test_forward_runs_when_initialized_from_params(self):
        # This code just passes things off to `transformers`, so we only have a very simple
        # test.
        params = Params({"model_name": "bert-base-uncased"})
        embedder = PretrainedTransformerEmbedder.from_params(params)
        token_ids = torch.randint(0, 100, (1, 4))
        mask = torch.randint(0, 2, (1, 4))
        output = embedder(token_ids=token_ids, mask=mask)
        assert tuple(output.size()) == (1, 4, 768)

    def test_end_to_end(self):
        tokenizer = PretrainedTransformerTokenizer(model_name="bert-base-uncased")
        token_indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased")

        sentence1 = "A, AllenNLP sentence."
        tokens1 = tokenizer.tokenize(sentence1)
        expected_tokens1 = ["[CLS]", "a", ",", "allen", "##nl", "##p", "sentence", ".", "[SEP]"]
        assert [t.text for t in tokens1] == expected_tokens1

        sentence2 = "AllenNLP is great"
        tokens2 = tokenizer.tokenize(sentence2)
        expected_tokens2 = ["[CLS]", "allen", "##nl", "##p", "is", "great", "[SEP]"]
        assert [t.text for t in tokens2] == expected_tokens2

        vocab = Vocabulary()

        params = Params(
            {
                "token_embedders": {
                    "bert": {"type": "pretrained_transformer", "model_name": "bert-base-uncased"}
                }
            }
        )
        token_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=params)

        instance1 = Instance({"tokens": TextField(tokens1, {"bert": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"bert": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]
        max_length = max(len(tokens1), len(tokens2))

        assert tokens["bert"]["token_ids"].shape == (2, max_length)

        assert tokens["bert"]["mask"].tolist() == [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
        ]

        # Attention mask
        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, 9, 768)

    def test_big_token_type_ids(self):
        token_embedder = PretrainedTransformerEmbedder("roberta-base")
        token_ids = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        mask = torch.ones_like(token_ids)
        type_ids = torch.zeros_like(token_ids)
        type_ids[1, 1] = 1
        with pytest.raises(ValueError):
            token_embedder(token_ids, mask, type_ids)
