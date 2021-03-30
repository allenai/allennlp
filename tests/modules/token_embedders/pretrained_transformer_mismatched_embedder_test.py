import pytest
import torch

from allennlp.common import Params
from allennlp.data import Token, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestPretrainedTransformerMismatchedEmbedder(AllenNlpTestCase):
    @pytest.mark.parametrize("train_parameters", [True, False])
    def test_end_to_end(self, train_parameters: bool):
        token_indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased")

        sentence1 = ["A", ",", "AllenNLP", "sentence", "."]
        sentence2 = ["AllenNLP", "is", "great"]
        tokens1 = [Token(word) for word in sentence1]
        tokens2 = [Token(word) for word in sentence2]

        vocab = Vocabulary()

        params = Params(
            {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": "bert-base-uncased",
                        "train_parameters": train_parameters,
                    }
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

        assert tokens["bert"]["offsets"].tolist() == [
            [[1, 1], [2, 2], [3, 5], [6, 6], [7, 7]],
            [[1, 3], [4, 4], [5, 5], [0, 0], [0, 0]],
        ]

        # Attention mask
        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, max(len(sentence1), len(sentence2)), 768)
        assert not torch.isnan(bert_vectors).any()
        assert bert_vectors.requires_grad == train_parameters

    def test_long_sequence_splitting_end_to_end(self):
        token_indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased", max_length=4)

        sentence1 = ["A", ",", "AllenNLP", "sentence", "."]
        sentence2 = ["AllenNLP", "is", "great"]
        tokens1 = [Token(word) for word in sentence1]
        tokens2 = [Token(word) for word in sentence2]

        vocab = Vocabulary()

        params = Params(
            {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": "bert-base-uncased",
                        "max_length": 4,
                    }
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

        assert tokens["bert"]["mask"].tolist() == [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
        assert tokens["bert"]["offsets"].tolist() == [
            [[1, 1], [2, 2], [3, 5], [6, 6], [7, 7]],
            [[1, 3], [4, 4], [5, 5], [0, 0], [0, 0]],
        ]

        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, max(len(sentence1), len(sentence2)), 768)
        assert not torch.isnan(bert_vectors).any()

    def test_token_without_wordpieces(self):
        token_indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased")

        sentence1 = ["A", "", "AllenNLP", "sentence", "."]
        sentence2 = ["AllenNLP", "", "great"]
        tokens1 = [Token(word) for word in sentence1]
        tokens2 = [Token(word) for word in sentence2]
        vocab = Vocabulary()
        params = Params(
            {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": "bert-base-uncased",
                    }
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

        assert tokens["bert"]["offsets"].tolist() == [
            [[1, 1], [-1, -1], [2, 4], [5, 5], [6, 6]],
            [[1, 3], [-1, -1], [4, 4], [0, 0], [0, 0]],
        ]

        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, max(len(sentence1), len(sentence2)), 768)
        assert not torch.isnan(bert_vectors).any()
        assert all(bert_vectors[0, 1] == 0)
        assert all(bert_vectors[1, 1] == 0)

    def test_exotic_tokens_no_nan_grads(self):
        token_indexer = PretrainedTransformerMismatchedIndexer("bert-base-uncased")

        sentence1 = ["A", "", "AllenNLP", "sentence", "."]
        sentence2 = ["A", "\uf732\uf730\uf730\uf733", "AllenNLP", "sentence", "."]

        tokens1 = [Token(word) for word in sentence1]
        tokens2 = [Token(word) for word in sentence2]
        vocab = Vocabulary()

        token_embedder = BasicTextFieldEmbedder(
            {"bert": PretrainedTransformerMismatchedEmbedder("bert-base-uncased")}
        )

        instance1 = Instance({"tokens": TextField(tokens1, {"bert": token_indexer})})
        instance2 = Instance({"tokens": TextField(tokens2, {"bert": token_indexer})})

        batch = Batch([instance1, instance2])
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]

        bert_vectors = token_embedder(tokens)
        test_loss = bert_vectors.mean()

        test_loss.backward()

        for name, param in token_embedder.named_parameters():
            grad = param.grad
            assert (grad is None) or (not torch.any(torch.isnan(grad)).item())
