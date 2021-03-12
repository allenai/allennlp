import pytest
import copy
import torch
from torch.testing import assert_allclose

from allennlp.common import Params, FromParams
from allennlp.common import cached_transformers

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertEmbeddings

from allennlp.common.testing import assert_equal_parameters
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    TransformerModule,
)
from allennlp.common.testing import AllenNlpTestCase

PARAMS_DICT = {
    "vocab_size": 20,
    "embedding_size": 5,
    "pad_token_id": 0,
    "max_position_embeddings": 3,
    "type_vocab_size": 2,
    "dropout": 0.5,
}


def get_modules(params_dict):
    modules = {}
    params = copy.deepcopy(params_dict)

    params["hidden_dropout_prob"] = params.pop("dropout")
    params["hidden_size"] = params.pop("embedding_size")

    # bert, roberta, electra self attentions have the same code.

    torch.manual_seed(1234)
    hf_module = BertEmbeddings(BertConfig(**params))
    modules["bert"] = hf_module

    albertparams = copy.deepcopy(params_dict)
    albertparams["hidden_dropout_prob"] = albertparams.pop("dropout")

    torch.manual_seed(1234)
    hf_module = AlbertEmbeddings(AlbertConfig(**albertparams))
    modules["albert"] = hf_module

    return modules


class TestTransformerEmbeddings(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {key: val for key, val in PARAMS_DICT.items()}

        params = Params(copy.deepcopy(self.params_dict))

        self.transformer_embeddings = TransformerEmbeddings.from_params(params)

    def test_can_construct_from_params(self):

        transformer_embeddings = self.transformer_embeddings.embeddings

        assert (
            transformer_embeddings.word_embeddings.num_embeddings == self.params_dict["vocab_size"]
        )
        assert (
            transformer_embeddings.word_embeddings.embedding_dim
            == self.params_dict["embedding_size"]
        )
        assert (
            transformer_embeddings.word_embeddings.padding_idx == self.params_dict["pad_token_id"]
        )

        assert (
            transformer_embeddings.position_embeddings.num_embeddings
            == self.params_dict["max_position_embeddings"]
        )
        assert (
            transformer_embeddings.position_embeddings.embedding_dim
            == self.params_dict["embedding_size"]
        )

        assert (
            transformer_embeddings.token_type_embeddings.num_embeddings
            == self.params_dict["type_vocab_size"]
        )
        assert (
            transformer_embeddings.token_type_embeddings.embedding_dim
            == self.params_dict["embedding_size"]
        )

        assert (
            self.transformer_embeddings.layer_norm.normalized_shape[0]
            == self.params_dict["embedding_size"]
        )

        assert self.transformer_embeddings.dropout.p == self.params_dict["dropout"]

    def test_sanity(self):
        class TextEmbeddings(TransformerModule, FromParams):
            def __init__(
                self,
                vocab_size: int,
                hidden_size: int,
                pad_token_id: int,
                max_position_embeddings: int,
                type_vocab_size: int,
                dropout: float,
            ):
                super().__init__()
                self.word_embeddings = torch.nn.Embedding(
                    vocab_size, hidden_size, padding_idx=pad_token_id
                )
                self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
                self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

                self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(
                self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
            ):
                if input_ids is not None:
                    input_shape = input_ids.size()
                else:
                    input_shape = inputs_embeds.size()[:-1]

                seq_length = input_shape[1]
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                if position_ids is None:
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)
                if token_type_ids is None:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

                if inputs_embeds is None:
                    inputs_embeds = self.word_embeddings(input_ids)
                position_embeddings = self.position_embeddings(position_ids)
                token_type_embeddings = self.token_type_embeddings(token_type_ids)

                embeddings = inputs_embeds + position_embeddings + token_type_embeddings
                embeddings = self.layer_norm(embeddings)
                embeddings = self.dropout(embeddings)
                return embeddings

        torch.manual_seed(23)
        text = TextEmbeddings(10, 5, 2, 3, 7, 0.0)
        torch.manual_seed(23)
        transformer = TransformerEmbeddings(10, 5, 2, 3, 7, 0.0)

        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])

        text_output = text.forward(input_ids, token_type_ids, position_ids)
        transformer_output = transformer.forward(input_ids, token_type_ids, position_ids)

        assert_allclose(text_output, transformer_output)

    def test_forward_runs_with_inputs(self):
        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])
        self.transformer_embeddings.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

    def test_output_size(self):
        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])
        params = copy.deepcopy(self.params_dict)
        params["output_size"] = 7
        params = Params(params)
        module = TransformerEmbeddings.from_params(params)
        output = module.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

        assert output.shape[-1] == 7

    def test_no_token_type_layer(self):
        params = copy.deepcopy(self.params_dict)
        params["type_vocab_size"] = 0
        params = Params(params)
        module = TransformerEmbeddings.from_params(params)

        assert len(module.embeddings) == 2

    @pytest.mark.parametrize(
        "pretrained_name",
        [
            "bert-base-uncased",
            "albert-base-v2",
        ],
    )
    def test_loading_from_pretrained_weights_using_model_name(self, pretrained_name):
        pretrained_module = cached_transformers.get(pretrained_name, False).embeddings
        module = TransformerEmbeddings.from_pretrained_module(pretrained_name)
        mapping = {
            val: key
            for key, val in module._construct_default_mapping(
                pretrained_module, "huggingface", {}
            ).items()
        }
        missing = assert_equal_parameters(pretrained_module, module, mapping=mapping)
        assert len(missing) == 0

    @pytest.mark.parametrize("module_name, hf_module", get_modules(PARAMS_DICT).items())
    def test_forward_against_huggingface_output(self, module_name, hf_module):
        input_ids = torch.tensor([[1, 2]])
        token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1]])

        torch.manual_seed(1234)
        embeddings = TransformerEmbeddings.from_pretrained_module(hf_module)

        torch.manual_seed(1234)
        embeddings = embeddings.eval()  # setting to eval mode to avoid non-deterministic dropout.
        output = embeddings.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

        torch.manual_seed(1234)
        hf_module = hf_module.eval()  # setting to eval mode to avoid non-deterministic dropout.
        hf_output = hf_module.forward(
            input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )

        assert torch.allclose(output, hf_output)


class TestImageFeatureEmbeddings(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        self.params_dict = {"feature_size": 3, "embedding_size": 5, "dropout": 0.1}

        params = Params(copy.deepcopy(self.params_dict))

        self.img_embeddings = ImageFeatureEmbeddings.from_params(params)

    def test_can_construct_from_params(self):
        assert (
            self.img_embeddings.embeddings.image_embeddings.in_features
            == self.params_dict["feature_size"]
        )
        assert (
            self.img_embeddings.embeddings.image_embeddings.out_features
            == self.params_dict["embedding_size"]
        )
        assert (
            self.img_embeddings.embeddings.location_embeddings.out_features
            == self.params_dict["embedding_size"]
        )
        assert self.img_embeddings.dropout.p == self.params_dict["dropout"]

    def test_forward_runs_with_inputs(self):
        batch_size = 2
        feature_dim = self.params_dict["feature_size"]
        image_feature = torch.randn(batch_size, feature_dim)
        image_location = torch.randn(batch_size, 4)
        self.img_embeddings.forward(image_feature, image_location)

    def test_sanity(self):
        class OldImageFeatureEmbeddings(TransformerModule, FromParams):
            """Construct the embeddings from image, spatial location (omit now) and
            token_type embeddings.
            """

            def __init__(self, feature_size: int, embedding_size: int, dropout: float = 0.0):
                super().__init__()

                self.image_embeddings = torch.nn.Linear(feature_size, embedding_size)
                self.image_location_embeddings = torch.nn.Linear(4, embedding_size, bias=False)
                self.layer_norm = torch.nn.LayerNorm(embedding_size, eps=1e-12)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, image_feature: torch.Tensor, image_location: torch.Tensor):
                img_embeddings = self.image_embeddings(image_feature)
                loc_embeddings = self.image_location_embeddings(image_location)
                embeddings = self.layer_norm(img_embeddings + loc_embeddings)
                embeddings = self.dropout(embeddings)

                return embeddings

        torch.manual_seed(23)
        old = OldImageFeatureEmbeddings(**self.params_dict)
        torch.manual_seed(23)
        now = ImageFeatureEmbeddings(**self.params_dict)

        batch_size = 2

        image_feature = torch.randn(batch_size, self.params_dict["feature_size"])
        image_location = torch.randn(batch_size, 4)

        torch.manual_seed(23)
        old_output = old.forward(image_feature, image_location)
        torch.manual_seed(23)
        now_output = now.forward(image_feature, image_location)

        assert_allclose(old_output, now_output)
