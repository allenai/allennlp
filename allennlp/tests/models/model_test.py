import torch

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.models import load_archive


class TestModel(AllenNlpTestCase):
    def test_extend_embedder_vocab(self):
        model_archive = str(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        trained_model = load_archive(model_archive).model
        # config_file used: str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        # dataset used: str(self.FIXTURES_ROOT / 'data' / 'snli.jsonl')

        original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(original_weight.shape) == (24, 300)

        counter = {"tokens": {"seahorse": 1}}  # 'seahorse' is extra token in snli2.jsonl
        trained_model.vocab._extend(counter)
        trained_model.extend_embedder_vocab()

        extended_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(extended_weight.shape) == (25, 300)

        assert torch.all(original_weight == extended_weight[:24, :])
