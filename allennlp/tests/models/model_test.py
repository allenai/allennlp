# pylint: disable=protected-access
import json

import torch

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

    def test_inspection_dict(self):
        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        parameters_inspection = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'parameters_inspection.json')
        modules_inspection = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'modules_inspection.json')

        model = load_archive(model_archive).model

        with open(parameters_inspection) as file:
            parameters_inspection_dict = json.load(file)

        with open(modules_inspection) as file:
            modules_inspection_dict = json.load(file)

        assert modules_inspection_dict == model.inspection_dict(inspect="modules", quite=True)
        assert parameters_inspection_dict == model.inspection_dict(inspect="parameters", quite=True)
