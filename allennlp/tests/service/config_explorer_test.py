# pylint: disable=no-self-use,invalid-name,line-too-long
import json
import os

import flask
import flask.testing

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.service.config_explorer import make_app


class TestConfigExplorer(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        app = make_app()
        app.testing = True
        self.client = app.test_client()

    def test_root_config(self):
        response = self.client.get('/')
        html = response.get_data().decode('utf-8')

        assert "allennlp.data.vocabulary.Vocabulary" in html
        assert "/?class=allennlp.data.vocabulary.Vocabulary" in html

    def test_choices(self):
        response = self.client.get('/?class=allennlp.data.dataset_readers.dataset_reader.DatasetReader')
        html = response.get_data().decode('utf-8')

        assert "allennlp.data.dataset_readers.semantic_role_labeling.SrlReader" in html
        assert "/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader" in html

    def test_subclass(self):
        response = self.client.get('/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        html = response.get_data().decode('utf-8')

        assert '"type": "srl"' in html
        assert '// "token_indexers"' in html

    def test_torch_class(self):
        response = self.client.get('/?class=torch.optim.rmsprop.RMSprop')
        html = response.get_data().decode('utf-8')

        assert '"type": "rmsprop"' in html
        assert '// "weight_decay"' in html

    def test_rnn_hack(self):
        response = self.client.get('/?class=torch.nn.modules.rnn.LSTM')
        html = response.get_data().decode('utf-8')

        assert '"type": "lstm"' in html
        assert '// "batch_first"' in html
