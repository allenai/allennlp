import copy
import torch

from allennlp.common import Params
from allennlp.modules.transformer import TransformerEmbeddings
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

        self.transformer_embeddings = TransformerEmbeddings.from_params(params)

    def test_can_construct_from_params(self):

        transformer_embeddings = self.transformer_embeddings.embeddings.embeddings

        assert transformer_embeddings[0].num_embeddings == self.params_dict["vocab_size"]
        assert transformer_embeddings[0].embedding_dim == self.params_dict["hidden_size"]
        assert transformer_embeddings[0].padding_idx == self.params_dict["pad_token_id"]

        assert (
            transformer_embeddings[1].num_embeddings == self.params_dict["max_position_embeddings"]
        )
        assert transformer_embeddings[1].embedding_dim == self.params_dict["hidden_size"]

        assert transformer_embeddings[2].num_embeddings == self.params_dict["type_vocab_size"]
        assert transformer_embeddings[2].embedding_dim == self.params_dict["hidden_size"]

        assert (
            self.transformer_embeddings.embeddings.layer_norm.normalized_shape[0]
            == self.params_dict["hidden_size"]
        )

        assert self.transformer_embeddings.embeddings.dropout.p == self.params_dict["dropout"]

    def test_forward_runs_with_inputs(self):
        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])
        self.transformer_embeddings.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )
