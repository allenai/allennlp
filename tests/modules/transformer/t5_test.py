import pytest
from transformers.models import t5 as hf_t5

from allennlp.modules.transformer.t5 import T5
from allennlp.nn.parallel import FairScaleFsdpAccelerator
from allennlp.common.testing import run_distributed_test, requires_multi_gpu


@pytest.mark.skip("takes too long in CI")
@pytest.mark.parametrize(
    "pretrained_model_name",
    [
        "t5-base",
        #  "t5-large",  # Takes WAY too long in CI
    ],
)
def test_create_t5_from_pretrained(pretrained_model_name: str):
    model = T5.from_pretrained_module(pretrained_model_name)
    # Make sure weights are tied.
    assert id(model.token_embeddings.weight) == id(model.lm_head.weight)


@pytest.fixture(scope="module")
def model_name():
    return "t5-small"


@pytest.fixture(scope="module")
def model(model_name):
    model = T5.from_pretrained_module(model_name).eval()
    model.beam_search.max_steps = 5
    return model


@pytest.fixture(scope="module")
def tokenizer(model_name):
    return hf_t5.T5Tokenizer.from_pretrained(model_name)


@pytest.fixture(scope="module")
def hf_model(model_name):
    return hf_t5.T5ForConditionalGeneration.from_pretrained(model_name).eval()


def test_t5_forward_loss(
    model: T5,
    hf_model: hf_t5.T5ForConditionalGeneration,
    tokenizer: hf_t5.T5Tokenizer,
):
    """
    Make sure loss and generation match the implementation from HF.
    """
    input_ids = tokenizer(
        ["The <extra_id_0> walks in <extra_id_1> park", "The <extra_id_0> barked"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert input_ids.tolist() == [
        [37, 32099, 10681, 16, 32098, 2447, 1],
        [37, 32099, 1207, 5100, 1, 0, 0],
    ]

    attention_mask = ~(input_ids == 0)

    labels = tokenizer(
        ["<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", "<extra_id_0> dog"],
        return_tensors="pt",
        padding=True,
    ).input_ids
    assert labels.tolist() == [
        [32099, 5295, 1782, 32098, 8, 32097, 1],
        [32099, 1782, 1, 0, 0, 0, 0],
    ]

    decoder_attention_mask = ~(labels == 0)

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )
    hf_outputs = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert outputs.loss == hf_outputs.loss
    assert (outputs.logits == hf_outputs.logits).all()


def test_t5_forward_beam_search(model: T5, tokenizer: hf_t5.T5Tokenizer):
    """
    Make sure beam search generates reasonable results, and that we get the same results
    for a given input, regardless of whether we run it on its own or part of a batch.
    """

    def run_beam_search(sents):
        input_ids = tokenizer(sents, return_tensors="pt", padding=True).input_ids
        outputs = model(input_ids)
        preds = [tokenizer.decode(preds[0]) for preds in outputs.predictions]
        probs = [round(float(p[0]), 4) for p in outputs.predicted_log_probs.exp()]
        return preds, probs

    s1 = "translate English to German: That is good"
    s2 = (
        "mnli premise: The Old One always comforted Ca'daan, except today. "
        "hypothesis: Ca'daan knew the Old One very well."
    )

    s1_pred, s1_prob = run_beam_search([s1])
    assert s1_pred == ["Das ist gut.</s>"]
    assert s1_prob == [0.5645]

    s2_pred, s2_prob = run_beam_search([s2])
    assert s2_pred == ["neutral</s> </s> </s> </s>"]
    assert s2_prob == [0.3992]

    combined_preds, combined_probs = run_beam_search([s1, s2])
    assert combined_preds == s1_pred + s2_pred
    assert combined_probs == s1_prob + s2_prob


def _test_distributed_load_state_dict(global_rank, world_size, gpu_id):
    T5.from_pretrained_module(
        "t5-small",
        ddp_accelerator=FairScaleFsdpAccelerator(
            local_rank=global_rank, world_size=world_size, cuda_device=gpu_id
        ),
    )


@requires_multi_gpu
def test_distributed_load_state_dict():
    run_distributed_test([0, 1], func=_test_distributed_load_state_dict)
