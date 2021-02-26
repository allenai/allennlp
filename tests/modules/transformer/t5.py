from allennlp.modules.transformer.t5 import *


def test_create_default_t5():
    t5 = T5ForConditionalGeneration()


def test_create_t5_with_weights():
    t5 = T5ForConditionalGeneration.from_pretrained_module("t5-large")


def test_create_t5_with_different_size_than_default():
    t5 = T5ForConditionalGeneration.from_pretrained_module("t5-small")
