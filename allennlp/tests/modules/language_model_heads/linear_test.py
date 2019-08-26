# pylint: disable=invalid-name,no-self-use,protected-access
import torch
from numpy.testing import assert_almost_equal
import numpy

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.language_model_heads import LanguageModelHead, LinearLanguageModelHead


class TestLinearLanguageModelHead(AllenNlpTestCase):
    def test_can_init_and_run(self):
        # The LM head code reads a module from somewhere else; we're basically just testing here
        # that we can initialize the expected model `from_params`.
        vocab = Vocabulary()
        # Using "tags" to avoid padding and unk tokens for the test.
        vocab.add_tokens_to_namespace(['this', 'is', 'a', 'test'], namespace='tags')
        head = LanguageModelHead.from_params(Params({"type": "linear",
                                                     "input_dim": 5,
                                                     "vocab_namespace": "tags"}),
                                             vocab=vocab)
        assert isinstance(head, LinearLanguageModelHead)
        assert head.get_input_dim() == 5
        assert head.get_output_dim() == 4
        tensor = torch.rand(1, 5)
        logits = head(tensor)
        assert tuple(logits.size()) == (1, 4)
