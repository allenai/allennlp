# pylint: disable=invalid-name,protected-access
from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


import copy
import os

from numpy.testing import assert_allclose
import torch

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive

from ipdb import set_trace

class WordLMTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'ptb.txt')

    def test_word_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_mismatching_dimensions_throws_configuration_error(self):

        params = Params.from_file(self.param_file)
        params["model"]["text_field_embedder"]["tokens"]["embedding_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

        params = Params.from_file(self.param_file)
        params["model"]["proj"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

if __name__ == '__main__':

    tmp = WordLMTest()
    tmp.setUp()
    tmp.test_word_lm_can_train_save_and_load()