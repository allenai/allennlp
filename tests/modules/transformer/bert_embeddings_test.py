import copy
import torch

import pytest

from allennlp.common import Params
from allennlp.modules.transformer import BertEmbeddings
from allennlp.common.testing import AllenNlpTestCase


class TestBertEmbeddings(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {
            "vocab_size": 20,
            "hidden_size": 5,
            "pad_token_id": 0,
            "max_position_embeddings": 3,
            "type_vocab_size": 2,
            "dropout": 0.0,
        }

        params = Params(copy.deepcopy(self.params_dict))

        self.bert_embeddings = BertEmbeddings.from_params(params)

    def test_can_construct_from_params(self):

        bert_embeddings = self.bert_embeddings

        assert bert_embeddings.word_embeddings.num_embeddings == self.params_dict["vocab_size"]
        assert bert_embeddings.word_embeddings.embedding_dim == self.params_dict["hidden_size"]
        assert bert_embeddings.word_embeddings.padding_idx == self.params_dict["pad_token_id"]

        assert (
            bert_embeddings.position_embeddings.num_embeddings
            == self.params_dict["max_position_embeddings"]
        )
        assert bert_embeddings.position_embeddings.embedding_dim == self.params_dict["hidden_size"]

        assert (
            bert_embeddings.token_type_embeddings.num_embeddings
            == self.params_dict["type_vocab_size"]
        )
        assert (
            bert_embeddings.token_type_embeddings.embedding_dim == self.params_dict["hidden_size"]
        )

        assert bert_embeddings.layer_norm.normalized_shape[0] == self.params_dict["hidden_size"]

        assert bert_embeddings.dropout.p == self.params_dict["dropout"]

    def test_forward_runs_with_input_embeds(self):

        self.bert_embeddings.forward(input_embeds=torch.tensor([[[1, 2, 3, 4, 5]]]))

    def test_forward_fails_with_no_input_ids_or_input_embeds(self):

        with pytest.raises(AssertionError):
            self.bert_embeddings.forward()

    def test_forward_runs_with_inputs(self):
        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])
        self.bert_embeddings.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )
