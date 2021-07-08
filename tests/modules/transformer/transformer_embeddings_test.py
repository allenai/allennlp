import copy

import pytest
import torch
from torch.testing import assert_allclose
from transformers import AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import AlbertEmbeddings

from allennlp.common import Params, FromParams
from allennlp.modules.transformer import (
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
    TransformerModule,
)


PARAMS_DICT = {
    "vocab_size": 20,
    "embedding_size": 5,
    "pad_token_id": 0,
    "max_position_embeddings": 3,
    "type_vocab_size": 2,
    "dropout": 0.5,
}


@pytest.fixture
def params_dict():
    return copy.deepcopy(PARAMS_DICT)


@pytest.fixture
def params(params_dict):
    return Params(params_dict)


@pytest.fixture
def transformer_embeddings(params):
    return TransformerEmbeddings.from_params(params.duplicate())


def test_can_construct_from_params(params_dict, transformer_embeddings):
    embeddings = transformer_embeddings.embeddings
    assert embeddings.word_embeddings.num_embeddings == params_dict["vocab_size"]
    assert embeddings.word_embeddings.embedding_dim == params_dict["embedding_size"]
    assert embeddings.word_embeddings.padding_idx == params_dict["pad_token_id"]

    assert embeddings.position_embeddings.num_embeddings == params_dict["max_position_embeddings"]
    assert embeddings.position_embeddings.embedding_dim == params_dict["embedding_size"]

    assert embeddings.token_type_embeddings.num_embeddings == params_dict["type_vocab_size"]
    assert embeddings.token_type_embeddings.embedding_dim == params_dict["embedding_size"]

    assert transformer_embeddings.layer_norm.normalized_shape[0] == params_dict["embedding_size"]

    assert transformer_embeddings.dropout.p == params_dict["dropout"]


def test_sanity():
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
    transformer = TransformerEmbeddings(10, 5, 2, 3, None, 7, 0.0)

    input_ids = torch.tensor([[1, 2]])
    token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]])

    text_output = text(input_ids, token_type_ids, position_ids)
    transformer_output = transformer(input_ids, token_type_ids, position_ids)

    assert_allclose(text_output, transformer_output)


def test_forward_runs_with_inputs(transformer_embeddings):
    input_ids = torch.tensor([[1, 2]])
    token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]])
    transformer_embeddings(
        input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
    )


def test_output_size(params):
    input_ids = torch.tensor([[1, 2]])
    token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]])
    params["output_size"] = 7
    module = TransformerEmbeddings.from_params(params)
    output = module(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    assert output.shape[-1] == 7


def test_no_token_type_layer(params):
    params["type_vocab_size"] = 0
    module = TransformerEmbeddings.from_params(params)
    assert len(module.embeddings) == 2


@pytest.mark.parametrize(
    "pretrained_name",
    [
        "bert-base-cased",
        "epwalsh/bert-xsmall-dummy",
    ],
)
def test_loading_from_pretrained_module(pretrained_name):
    TransformerEmbeddings.from_pretrained_module(pretrained_name)


def test_loading_albert():
    """
    Albert is a special case because it includes a Linear layer in the encoder
    that maps the embeddings to the encoder hidden size, but we include this linear
    layer within our embedding layer.
    """
    transformer_embedding = TransformerEmbeddings.from_pretrained_module(
        "albert-base-v2",
    )
    albert = AutoModel.from_pretrained("albert-base-v2")
    assert_allclose(
        transformer_embedding.embeddings.word_embeddings.weight.data,
        albert.embeddings.word_embeddings.weight.data,
    )
    assert_allclose(
        transformer_embedding.linear_transform.weight.data,
        albert.encoder.embedding_hidden_mapping_in.weight.data,
    )


def get_modules():
    params = copy.deepcopy(PARAMS_DICT)

    params["hidden_dropout_prob"] = params.pop("dropout")
    params["hidden_size"] = params.pop("embedding_size")

    # bert, roberta, electra self attentions have the same code.

    torch.manual_seed(1234)
    yield "bert", BertEmbeddings(BertConfig(**params))

    albertparams = copy.deepcopy(PARAMS_DICT)
    albertparams["hidden_dropout_prob"] = albertparams.pop("dropout")

    torch.manual_seed(1234)
    yield "albert", AlbertEmbeddings(AlbertConfig(**albertparams))


@pytest.mark.parametrize("module_name, hf_module", get_modules())
def test_forward_against_huggingface_output(transformer_embeddings, module_name, hf_module):
    input_ids = torch.tensor([[1, 2]])
    token_type_ids = torch.tensor([[1, 0]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]])

    state_dict = transformer_embeddings._get_mapped_state_dict(hf_module.state_dict())
    if "position_ids" in state_dict:
        del state_dict["position_ids"]
    transformer_embeddings.load_state_dict(state_dict)

    torch.manual_seed(1234)
    transformer_embeddings = (
        transformer_embeddings.eval()
    )  # setting to eval mode to avoid non-deterministic dropout.
    output = transformer_embeddings(
        input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
    )

    torch.manual_seed(1234)
    hf_module = hf_module.eval()  # setting to eval mode to avoid non-deterministic dropout.
    hf_output = hf_module(
        input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids
    )

    assert torch.allclose(output, hf_output)


@pytest.fixture
def image_params_dict():
    return {"feature_size": 3, "embedding_size": 5, "dropout": 0.1}


@pytest.fixture
def image_params(image_params_dict):
    return Params(image_params_dict)


@pytest.fixture
def image_embeddings(image_params):
    return ImageFeatureEmbeddings.from_params(image_params.duplicate())


def test_can_construct_image_embeddings_from_params(image_embeddings, image_params_dict):
    assert (
        image_embeddings.embeddings.image_embeddings.in_features
        == image_params_dict["feature_size"]
    )
    assert (
        image_embeddings.embeddings.image_embeddings.out_features
        == image_params_dict["embedding_size"]
    )
    assert (
        image_embeddings.embeddings.location_embeddings.out_features
        == image_params_dict["embedding_size"]
    )
    assert image_embeddings.dropout.p == image_params_dict["dropout"]


def test_image_embedding_forward_runs_with_inputs(image_embeddings, image_params_dict):
    batch_size = 2
    feature_dim = image_params_dict["feature_size"]
    image_feature = torch.randn(batch_size, feature_dim)
    image_location = torch.randn(batch_size, 4)
    image_embeddings(image_feature, image_location)


def test_image_embeddings_sanity(image_params_dict):
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
    old = OldImageFeatureEmbeddings(**image_params_dict)
    torch.manual_seed(23)
    now = ImageFeatureEmbeddings(**image_params_dict)

    batch_size = 2

    image_feature = torch.randn(batch_size, image_params_dict["feature_size"])
    image_location = torch.randn(batch_size, 4)

    torch.manual_seed(23)
    old_output = old(image_feature, image_location)
    torch.manual_seed(23)
    now_output = now(image_feature, image_location)

    assert_allclose(old_output, now_output)
