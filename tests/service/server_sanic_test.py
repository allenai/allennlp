# pylint: disable=no-self-use,invalid-name
import json
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_sanic import make_app

TEST_ARCHIVE_FILES = {
        'machine-comprehension': 'tests/fixtures/bidaf/serialization/model.tar.gz',
        'semantic-role-labeling': 'tests/fixtures/srl/serialization/model.tar.gz',
        'textual-entailment': 'tests/fixtures/decomposable_attention/serialization/model.tar.gz'
}

class TestApp(AllenNlpTestCase):

    client = None

    def setUp(self):
        super(TestApp, self).setUp()
        if self.client is None:
            app = make_app()
            app.predictors = {
                    name: Predictor.from_archive(load_archive(archive_file))
                    for name, archive_file in TEST_ARCHIVE_FILES.items()
            }

            app.testing = True
            self.client = app.test_client

    def tearDown(self):
        super(TestApp, self).tearDown()
        try:
            os.remove('access.log')
            os.remove('error.log')
        except FileNotFoundError:
            pass

    def test_list_models(self):
        _, response = self.client.get("/models")
        data = json.loads(response.text)
        assert "machine-comprehension" in set(data["models"])

    def test_unknown_model(self):
        _, response = self.client.post("/predict/bogus_model",
                                       json={"input": "broken"})
        assert response.status == 400
        assert "unknown model" in response.text and "bogus_model" in response.text

    def test_machine_comprehension(self):
        _, response = self.client.post("/predict/machine-comprehension",
                                       json={"passage": "the super bowl was played in seattle",
                                             "question": "where was the super bowl played?"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "best_span" in results

    def test_textual_entailment(self):
        _, response = self.client.post("/predict/textual-entailment",
                                       json={"premise": "the super bowl was played in seattle",
                                             "hypothesis": "the super bowl was played in ohio"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "label_probs" in results

    def test_semantic_role_labeling(self):
        _, response = self.client.post("/predict/semantic-role-labeling",
                                       json={"sentence": "the super bowl was played in seattle"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "verbs" in results
