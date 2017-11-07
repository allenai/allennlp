# pylint: disable=no-self-use,invalid-name
import copy
import json
import os
import pathlib
from collections import defaultdict

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_sanic import make_app
from allennlp.service.db import InMemoryDemoDatabase

TEST_ARCHIVE_FILES = {
        'machine-comprehension': 'tests/fixtures/bidaf/serialization/model.tar.gz',
        'semantic-role-labeling': 'tests/fixtures/srl/serialization/model.tar.gz',
        'textual-entailment': 'tests/fixtures/decomposable_attention/serialization/model.tar.gz'
}

PREDICTORS = {
        name: Predictor.from_archive(load_archive(archive_file),
                                     predictor_name=name)
        for name, archive_file in TEST_ARCHIVE_FILES.items()
}


class CountingPredictor(Predictor):
    """
    bogus predictor that just returns a copy of its inputs
    and also counts how many times it was called with a given input
    """
    # pylint: disable=abstract-method
    def __init__(self):                 # pylint: disable=super-init-not-called
        self.calls = defaultdict(int)

    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        key = json.dumps(inputs)
        self.calls[key] += 1
        return copy.deepcopy(inputs)

class TestSanic(AllenNlpTestCase):

    client = None

    def setUp(self):
        super().setUp()
        # Create index.html in TEST_DIR
        pathlib.Path(os.path.join(self.TEST_DIR, 'index.html')).touch()

        if self.client is None:

            self.app = make_app(build_dir=self.TEST_DIR)
            self.app.predictors = PREDICTORS
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
        app = server_sanic.make_app(build_dir=self.TEST_DIR)
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

    def test_missing_static_dir(self):
        fake_dir = os.path.join(self.TEST_DIR, '/this/directory/does/not/exist')

        with self.assertRaises(SystemExit) as cm:
            make_app(fake_dir)
            assert cm.code == -1  # pylint: disable=no-member

    def test_permalinks_fail_gracefully_with_no_database(self):
        app = make_app(build_dir=self.TEST_DIR)
        predictor = CountingPredictor()
        app.predictors = {"counting": predictor}
        app.testing = True
        client = app.test_client

        # Make a prediction, no permalinks.
        data = {"some": "input"}
        _, response = client.post("/predict/counting", json=data)

        assert response.status == 200

        # With permalinks not enabled, the result shouldn't contain a slug.
        result = json.loads(response.text)
        assert "slug" not in result

        # With permalinks not enabled, a post to the /permadata endpoint should be a 400.
        _, response = self.client.post("/permadata", json={"slug": "someslug"})
        assert response.status == 400

    def test_permalinks_work(self):
        db = InMemoryDemoDatabase()
        app = make_app(build_dir=self.TEST_DIR, demo_db=db)
        predictor = CountingPredictor()
        app.predictors = {"counting": predictor}
        app.testing = True
        client = app.test_client

        data = {"some": "input"}
        _, response = client.post("/predict/counting", json=data)

        assert response.status == 200
        result = json.loads(response.text)
        slug = result.get("slug")
        assert slug is not None

        print("db data", db.data)

        _, response = client.post("/permadata", json={"slug": "not the right slug"})
        assert response.status == 400

        _, response = client.post("/permadata", json={"slug": slug})
        assert response.status == 200
        result2 = json.loads(response.text)
        assert set(result2.keys()) == {"modelName", "requestData", "responseData"}
        assert result2["modelName"] == "counting"
        assert result2["requestData"] == data
        assert result2["responseData"] == result
