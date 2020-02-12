import torch

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.language_model_heads import BertLanguageModelHead, LanguageModelHead


class TestBertLanguageModelHead(AllenNlpTestCase):
    def test_can_init_and_run(self):
        # The LM head code reads a module from somewhere else; we're basically just testing here
        # that we can initialize the expected model `from_params`.
        head = LanguageModelHead.from_params(
            Params({"type": "bert", "model_name": "bert-base-uncased"})
        )
        assert isinstance(head, BertLanguageModelHead)
        assert head.get_input_dim() == 768
        assert head.get_output_dim() == 30522
        tensor = torch.rand(1, 768)
        logits = head(tensor)
        assert tuple(logits.size()) == (1, 30522)
