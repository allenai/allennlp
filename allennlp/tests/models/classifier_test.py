import numpy as np
import torch
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase, AllenNlpTestCase
from allennlp.data.dataset import Batch
from allennlp.models import Classifier


class TestClassifiers(ModelTestCase):
    def setUp(self):
        super().setUp()

    def test_logistic_regression_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'classifier' / 'experiment_logistic_regression.json',
                          AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_seq2vec_clf_with_vae_token_embedder_can_train_save_and_load(self):
        self.set_up_model(AllenNlpTestCase.FIXTURES_ROOT / 'classifier' / 'experiment_seq2vec.json',
                          AllenNlpTestCase.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_train.jsonl")
        self.ensure_model_can_train_save_and_load(self.param_file)
