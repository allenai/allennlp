import torch

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.models import load_archive


class TestModel(AllenNlpTestCase):
    def test_extend_embedder_vocab(self):
        model_archive = str(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        trained_model = load_archive(model_archive).model

        original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(original_weight.shape) == (213, 10)

        counter = {"tokens": {"unawarded": 1}}
        trained_model.vocab._extend(counter)
        trained_model.extend_embedder_vocab()

        extended_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(extended_weight.shape) == (214, 10)

        assert torch.all(original_weight == extended_weight[:213, :])
