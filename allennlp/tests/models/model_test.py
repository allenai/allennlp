# pylint: disable=protected-access
import torch
import copy

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.models import load_archive

class TestModel(AllenNlpTestCase):
    def test_extend_embedder_vocab(self):
        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model
        # config_file used: str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        # dataset used: str(self.FIXTURES_ROOT / 'data' / 'snli.jsonl')

        original_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(original_weight.shape) == (24, 300)

        vocab = trained_model.vocab
        counter = {"tokens": {"seahorse": 1}} # 'seahorse' is extra token in snli2.jsonl
        vocab._extend(counter)
        trained_model.extend_embedder_vocab(vocab)

        extended_weight = trained_model._text_field_embedder.token_embedder_tokens.weight
        assert tuple(extended_weight.shape) == (25, 300)

        assert torch.all(original_weight == extended_weight[:24, :])

    def test_get_module_path(self):
        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        model = load_archive(model_archive).model

        module_path_1 = '_aggregate_feedforward._linear_layers'
        module_path_2 = '_compare_feedforward._module._dropout.0'

        modules_dict = {module_path: module for module_path, module in model.named_modules()}

        module_1 = modules_dict[module_path_1]
        module_2 = modules_dict[module_path_2]

        assert module_path_1 == model.get_module_path(module_1)
        assert module_path_2 == model.get_module_path(module_2)
