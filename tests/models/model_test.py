import torch
import pytest

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.models import load_archive, Model
from allennlp.nn.regularizers import RegularizerApplicator


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

    def test_get_regularization_penalty(self):
        class FakeModel(Model):
            def forward(self, **kwargs):
                return {}

        class FakeRegularizerApplicator(RegularizerApplicator):
            def __call__(self, module):
                return 2.0

        with pytest.raises(RuntimeError):
            regularizer = FakeRegularizerApplicator()
            model = FakeModel(None, regularizer)
            model.get_regularization_penalty()
