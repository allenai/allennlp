import pytest

from allennlp.modules.transformer.t5 import T5ForConditionalGeneration


def test_create_default_t5():
    T5ForConditionalGeneration()


@pytest.mark.skip(reason="Not implemented yet")
def test_create_t5_large_from_pretrained():
    T5ForConditionalGeneration.from_pretrained_module("t5-large")


def test_create_t5_small_from_pretrained():
    T5ForConditionalGeneration.from_pretrained_module("t5-small")
