import torch

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.language_model_heads import Gpt2LanguageModelHead, LanguageModelHead


class TestGpt2LanguageModelHead(AllenNlpTestCase):
    def test_can_init_and_run(self):
        # The LM head code reads a module from somewhere else; we're basically just testing here
        # that we can initialize the expected model `from_params`.
        head = LanguageModelHead.from_params(Params({"type": "gpt2", "model_name": "gpt2"}))
        assert isinstance(head, Gpt2LanguageModelHead)
        assert head.get_input_dim() == 768
        assert head.get_output_dim() == 50257
        tensor = torch.rand(1, 768)
        logits = head(tensor)
        assert tuple(logits.size()) == (1, 50257)
