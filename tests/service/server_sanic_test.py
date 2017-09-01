# pylint: disable=no-self-use,invalid-name
import json
import os
from collections import defaultdict

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_sanic import make_app

TEST_ARCHIVE_FILES = {
        'machine-comprehension': 'tests/fixtures/bidaf/serialization/model.tar.gz',
        'semantic-role-labeling': 'tests/fixtures/srl/serialization/model.tar.gz',
        'textual-entailment': 'tests/fixtures/decomposable_attention/serialization/model.tar.gz'
}

class CountingPredictor(Predictor):
    """
    bogus predictor that just returns its input as is
    and also counts how many times it was called with a given input
    """
    def __init__(self):                 # pylint: disable=super-init-not-called
        self.calls = defaultdict(int)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        key = json.dumps(inputs)
        self.calls[key] += 1
        return inputs

class TestSanic(AllenNlpTestCase):

    client = None

    def setUp(self):
        super().setUp()
        if self.client is None:
            self.app = make_app()
            self.app.predictors = {
                    name: Predictor.from_archive(load_archive(archive_file))
                    for name, archive_file in TEST_ARCHIVE_FILES.items()
            }

            self.app.testing = True
            self.client = self.app.test_client

    def tearDown(self):
        super().tearDown()
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

    def test_caching(self):
        predictor = CountingPredictor()
        data = {"input1": "this is input 1", "input2": 10}
        key = json.dumps(data)

        self.app.predictors["counting"] = predictor

        # call counts should be empty
        assert not predictor.calls

        _, response = self.client.post("/predict/counting", json=data)
        assert response.status == 200
        assert json.loads(response.text) == data

        # call counts should reflect the one call
        assert predictor.calls.get(key) == 1
        assert len(predictor.calls) == 1

        # make a different call
        noyes = {"no": "yes"}
        _, response = self.client.post("/predict/counting", json=noyes)
        assert response.status == 200
        assert json.loads(response.text) == noyes

        # call counts should reflect two calls
        assert predictor.calls[key] == 1
        assert predictor.calls[json.dumps(noyes)] == 1
        assert len(predictor.calls) == 2

        # repeated calls should come from cache and not hit the predictor
        for _ in range(3):
            _, response = self.client.post("/predict/counting", json=data)
            assert response.status == 200
            assert json.loads(response.text) == data

            # these should all be cached, so call counts should not be updated
            assert predictor.calls[key] == 1
            assert predictor.calls[json.dumps(noyes)] == 1
            assert len(predictor.calls) == 2

    def test_disable_caching(self):
        import allennlp.service.server_sanic as server_sanic
        server_sanic.CACHE_SIZE = 0

        predictor = CountingPredictor()
        app = server_sanic.make_app()
        app.predictors = {"counting": predictor}
        app.testing = True
        client = app.test_client

        data = {"input1": "this is input 1", "input2": 10}
        key = json.dumps(data)

        assert not predictor.calls

        for i in range(5):
            _, response = client.post("/predict/counting", json=data)
            assert response.status == 200
            assert json.loads(response.text) == data

            # cache is disabled, so call count should keep incrementing
            assert predictor.calls[key] == i + 1
            assert len(predictor.calls) == 1
