from allennlp.modules.transformer.t5 import *


def test_create_default_t5():
    t5 = T5ForConditionalGeneration()
