# pylint: disable=invalid-name,arguments-differ,abstract-method
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class TestMaskedLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'masked_language_model' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
