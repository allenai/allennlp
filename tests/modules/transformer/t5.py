import pytest
from transformers.models import t5 as hf_t5

from allennlp.modules.transformer.t5 import T5ForConditionalGeneration


@pytest.mark.skip(reason="Not implemented yet")
def test_create_t5_large_from_pretrained():
    T5ForConditionalGeneration.from_pretrained_module("t5-large")


@pytest.fixture(scope="module")
def model_name():
    return "t5-small"


@pytest.fixture(scope="module")
def model(model_name):
    model = T5ForConditionalGeneration.from_pretrained_module(model_name).eval()
    model.beam_search.max_steps = 5
    return model


@pytest.fixture(scope="module")
def tokenizer(model_name):
    return hf_t5.T5Tokenizer.from_pretrained(model_name)


@pytest.fixture(scope="module")
def hf_model(model_name):
    return hf_t5.T5ForConditionalGeneration.from_pretrained(model_name).eval()


@pytest.fixture
def input_ids(tokenizer):
    input_ids = tokenizer(
        ["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> barked"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert input_ids.tolist() == [
        [37, 32099, 10681, 16, 32098, 2447, 1],
        [37, 32099, 1207, 5100, 1, 0, 0],
    ]
    return input_ids


@pytest.fixture
def labels(tokenizer):
    labels = tokenizer(
        ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", "<extra_id_0> dog"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert labels.tolist() == [
        [32099, 5295, 1782, 32098, 8, 32097, 1],
        [32099, 1782, 1, 0, 0, 0, 0],
    ]
    return labels


@pytest.mark.skip(
    reason="This is covered by 'test_forward_loss', but can helpful for more fine-grained debugging"
)
def test_t5_encoder(
    model: T5ForConditionalGeneration, hf_model: hf_t5.T5ForConditionalGeneration, input_ids
):
    """
    Make sure loss and generation match the implementation from HF.
    """
    encoder_outputs = model.encoder(
        input_ids, output_attentions=True, output_all_hidden_states=True
    )
    hf_encoder_outputs = hf_model.encoder(
        input_ids, output_attentions=True, output_hidden_states=True
    )
    assert len(encoder_outputs.all_hidden_states) == len(hf_encoder_outputs.hidden_states)
    for layer, (hidden_state, hf_hidden_state) in enumerate(
        zip(encoder_outputs.all_hidden_states, hf_encoder_outputs.hidden_states)
    ):
        assert (hidden_state == hf_hidden_state).all(), f"Layer {layer} hidden states differ"


def test_t5_forward_loss(
    model: T5ForConditionalGeneration, hf_model: hf_t5.T5ForConditionalGeneration, input_ids, labels
):
    """
    Make sure loss and generation match the implementation from HF.
    """
    outputs = model(input_ids, labels=labels)
    hf_outputs = hf_model(input_ids=input_ids, labels=labels)
    assert outputs.loss == hf_outputs.loss
    assert (outputs.logits == hf_outputs.logits).all()
