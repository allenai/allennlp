import math
import pytest
import torch

from allennlp.common import Params, cached_transformers
from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.data import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder


class TestPretrainedTransformerEmbedder(AllenNlpTestCase):
    @classmethod
    def teardown_class(cls):
        cached_transformers._clear_caches()

    @requires_gpu
    def test_forward_runs_when_initialized_from_params(self):
        # This code just passes things off to `transformers`, so we only have a very simple
        # test.
        params = Params({"model_name": "bert-base-uncased"})
        embedder = PretrainedTransformerEmbedder.from_params(params).cuda()
        token_ids = torch.randint(0, 100, (1, 4))
        mask = torch.randint(0, 2, (1, 4)).bool()
        output = embedder(token_ids=token_ids.cuda(), mask=mask.cuda())
        assert tuple(output.size()) == (1, 4, 768)

    @pytest.mark.parametrize(
        "train_parameters, last_layer_only, gradient_checkpointing",
        [
            (train_parameters, last_layer_only, gradient_checkpointing)
            for train_parameters in {True, False}
            for last_layer_only in {True, False}
            for gradient_checkpointing in {True, False}
            if train_parameters
            or not gradient_checkpointing  # checkpointing only makes sense when we're actually training the layers
        ],
    )
    def test_end_to_end(
        self,
        train_parameters: bool,
        last_layer_only: bool,
        gradient_checkpointing: bool,
    ):
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
                    "bert": {
                        "type": "pretrained_transformer",
                        "model_name": "bert-base-uncased",
                        "train_parameters": train_parameters,
                        "last_layer_only": last_layer_only,
                        "gradient_checkpointing": gradient_checkpointing,
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
        max_length = max(len(tokens1), len(tokens2))

        assert tokens["bert"]["token_ids"].shape == (2, max_length)

        assert tokens["bert"]["mask"].tolist() == [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, False, False],
        ]

        # Attention mask
        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, 9, 768)
        assert bert_vectors.requires_grad == (train_parameters or not last_layer_only)

    @pytest.mark.parametrize(
        "train_parameters, last_layer_only, gradient_checkpointing",
        [
            (train_parameters, last_layer_only, gradient_checkpointing)
            for train_parameters in {True, False}
            for last_layer_only in {
                True
            }  # Huggingface T5 is totally different in the way it returns the
            # intermediate layers, and we don't support that.
            for gradient_checkpointing in {True, False}
            if train_parameters
            or not gradient_checkpointing  # checkpointing only makes sense when we're actually training the layers
        ],
    )
    def test_end_to_end_t5(
        self,
        train_parameters: bool,
        last_layer_only: bool,
        gradient_checkpointing: bool,
    ):
        tokenizer = PretrainedTransformerTokenizer(model_name="patrickvonplaten/t5-tiny-random")
        token_indexer = PretrainedTransformerIndexer(model_name="patrickvonplaten/t5-tiny-random")

        sentence1 = "A, AllenNLP sentence."
        tokens1 = tokenizer.tokenize(sentence1)
        expected_tokens1 = ["▁A", ",", "▁Allen", "N", "LP", "▁sentence", ".", "</s>"]
        assert [t.text for t in tokens1] == expected_tokens1

        sentence2 = "AllenNLP is great"
        tokens2 = tokenizer.tokenize(sentence2)
        expected_tokens2 = ["▁Allen", "N", "LP", "▁is", "▁great", "</s>"]
        assert [t.text for t in tokens2] == expected_tokens2

        vocab = Vocabulary()

        params = Params(
            {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer",
                        "model_name": "patrickvonplaten/t5-tiny-random",
                        "train_parameters": train_parameters,
                        "last_layer_only": last_layer_only,
                        "gradient_checkpointing": gradient_checkpointing,
                        "sub_module": "encoder",
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
        max_length = max(len(tokens1), len(tokens2))

        assert tokens["bert"]["token_ids"].shape == (2, max_length)

        assert tokens["bert"]["mask"].tolist() == [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, False, False],
        ]

        # Attention mask
        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, 8, 64)
        assert bert_vectors.requires_grad == (train_parameters or not last_layer_only)

    @requires_gpu
    def test_big_token_type_ids(self):
        token_embedder = PretrainedTransformerEmbedder("roberta-base").cuda()
        token_ids = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        mask = torch.ones_like(token_ids).bool()
        type_ids = torch.zeros_like(token_ids)
        type_ids[1, 1] = 1
        with pytest.raises(ValueError):
            token_embedder(token_ids.cuda(), mask.cuda(), type_ids.cuda())

    @requires_gpu
    def test_xlnet_token_type_ids(self):
        token_embedder = PretrainedTransformerEmbedder("xlnet-base-cased").cuda()
        token_ids = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        mask = torch.ones_like(token_ids).bool()
        type_ids = torch.zeros_like(token_ids)
        type_ids[1, 1] = 1
        token_embedder(token_ids.cuda(), mask.cuda(), type_ids.cuda())

    def test_long_sequence_splitting_end_to_end(self):
        # Mostly the same as the end_to_end test (except for adding max_length=4),
        # because we don't want this splitting behavior to change input/output format.

        tokenizer = PretrainedTransformerTokenizer(model_name="bert-base-uncased")
        token_indexer = PretrainedTransformerIndexer(model_name="bert-base-uncased", max_length=4)

        sentence1 = "A, AllenNLP sentence."
        tokens1 = tokenizer.tokenize(sentence1)
        sentence2 = "AllenNLP is great"
        tokens2 = tokenizer.tokenize(sentence2)

        vocab = Vocabulary()

        params = Params(
            {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer",
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
        max_length = max(len(tokens1), len(tokens2))

        # Adds n_segments * 2 special tokens
        segment_concat_length = int(math.ceil(max_length / 4)) * 2 + max_length
        assert tokens["bert"]["token_ids"].shape == (2, segment_concat_length)

        assert tokens["bert"]["mask"].tolist() == [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, False, False],
        ]
        assert tokens["bert"]["segment_concat_mask"].tolist() == [
            [True] * segment_concat_length,
            [True] * (segment_concat_length - 4) + [False] * 4,  # 4 is hard-coded length difference
        ]

        # Attention mask
        bert_vectors = token_embedder(tokens)
        assert bert_vectors.size() == (2, 9, 768)

    def test_fold_long_sequences(self):
        # Let's just say [PAD] is 0, [CLS] is 1, and [SEP] is 2
        token_ids = torch.LongTensor(
            [
                [1, 101, 102, 103, 104, 2, 1, 105, 106, 107, 108, 2, 1, 109, 2],
                [1, 201, 202, 203, 204, 2, 1, 205, 206, 207, 208, 2, 0, 0, 0],
                [1, 301, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )  # Shape: [3, 15]
        segment_concat_mask = (token_ids > 0).long()

        folded_token_ids = torch.LongTensor(
            [
                [1, 101, 102, 103, 104, 2],
                [1, 105, 106, 107, 108, 2],
                [1, 109, 2, 0, 0, 0],
                [1, 201, 202, 203, 204, 2],
                [1, 205, 206, 207, 208, 2],
                [0, 0, 0, 0, 0, 0],
                [1, 301, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        folded_segment_concat_mask = (folded_token_ids > 0).long()

        token_embedder = PretrainedTransformerEmbedder("bert-base-uncased", max_length=6)

        (
            folded_token_ids_out,
            folded_segment_concat_mask_out,
            _,
        ) = token_embedder._fold_long_sequences(token_ids, segment_concat_mask)
        assert (folded_token_ids_out == folded_token_ids).all()
        assert (folded_segment_concat_mask_out == folded_segment_concat_mask).all()

    def test_unfold_long_sequences(self):
        # Let's just say [PAD] is 0, [CLS] is xxx1, and [SEP] is xxx2
        # We assume embeddings are 1-dim and are the same as indices
        embeddings = torch.LongTensor(
            [
                [1001, 101, 102, 103, 104, 1002],
                [1011, 105, 106, 107, 108, 1012],
                [1021, 109, 1022, 0, 0, 0],
                [2001, 201, 202, 203, 204, 2002],
                [2011, 205, 206, 207, 208, 2012],
                [0, 0, 0, 0, 0, 0],
                [3001, 301, 3002, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ).unsqueeze(-1)
        mask = (embeddings > 0).long()

        unfolded_embeddings = torch.LongTensor(
            [
                [1001, 101, 102, 103, 104, 105, 106, 107, 108, 109, 1022],
                [2001, 201, 202, 203, 204, 205, 206, 207, 208, 2012, 0],
                [3001, 301, 3002, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).unsqueeze(-1)

        token_embedder = PretrainedTransformerEmbedder("bert-base-uncased", max_length=6)

        unfolded_embeddings_out = token_embedder._unfold_long_sequences(
            embeddings, mask, unfolded_embeddings.size(0), 15
        )
        assert (unfolded_embeddings_out == unfolded_embeddings).all()

    @requires_gpu
    def test_encoder_decoder_model(self):
        token_embedder = PretrainedTransformerEmbedder(
            "facebook/bart-large", sub_module="encoder"
        ).cuda()
        token_ids = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        mask = torch.ones_like(token_ids).bool()
        token_embedder(token_ids.cuda(), mask.cuda())

    def test_embeddings_resize(self):
        regular_token_embedder = PretrainedTransformerEmbedder("bert-base-cased")
        assert (
            regular_token_embedder.transformer_model.embeddings.word_embeddings.num_embeddings
            == 28996
        )
        tokenizer_kwargs = {"additional_special_tokens": ["<NEW_TOKEN>"]}
        enhanced_token_embedder = PretrainedTransformerEmbedder(
            "bert-base-cased", tokenizer_kwargs=tokenizer_kwargs
        )
        assert (
            enhanced_token_embedder.transformer_model.embeddings.word_embeddings.num_embeddings
            == 28997
        )

    def test_eval_mode(self):
        token_embedder = PretrainedTransformerEmbedder("epwalsh/bert-xsmall-dummy", eval_mode=True)
        assert token_embedder.training and not token_embedder.transformer_model.training

        class TrainableModule(torch.nn.Module):
            def __init__(self, fixed_module):
                super().__init__()
                self.fixed_module = fixed_module

        trainable = TrainableModule(token_embedder)
        assert (
            trainable.training
            and trainable.fixed_module.training
            and not trainable.fixed_module.transformer_model.training
        )

        trainable.train()
        assert not trainable.fixed_module.transformer_model.training
