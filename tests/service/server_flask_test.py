# pylint: disable=no-self-use,invalid-name
import copy
import json
import os
import pathlib
from collections import defaultdict

from flask import Response

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_flask import make_app
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

class TestFlask(AllenNlpTestCase):

    client = None

    def setUp(self):
        super().setUp()
        # Create index.html in TEST_DIR
        pathlib.Path(os.path.join(self.TEST_DIR, 'index.html')).touch()

        if self.client is None:

            self.app = make_app(build_dir=self.TEST_DIR)
            self.app.predictors = PREDICTORS
            self.app.testing = True
            self.client = self.app.test_client()

    def post_json(self, endpoint: str, data: JsonDict) -> Response:
        return self.client.post(endpoint,
                                content_type="application/json",
                                data=json.dumps(data))


    def tearDown(self):
        super().tearDown()
        try:
            os.remove('access.log')
            os.remove('error.log')
        except FileNotFoundError:
            pass

    def test_list_models(self):
        response = self.client.get("/models")
        data = json.loads(response.get_data())
        assert "machine-comprehension" in set(data["models"])

    def test_unknown_model(self):
        response = self.post_json("/predict/bogus_model",
                                  data={"input": "broken"})
        assert response.status_code == 400
        data = response.get_data()
        assert b"unknown model" in data and b"bogus_model" in data

    def test_machine_comprehension(self):
        response = self.post_json("/predict/machine-comprehension",
                                  data={"passage": "the super bowl was played in seattle",
                                        "question": "where was the super bowl played?"})

        assert response.status_code == 200
        results = json.loads(response.data)
        assert "best_span" in results

    def test_textual_entailment(self):
        response = self.post_json("/predict/textual-entailment",
                                  data={"premise": "the super bowl was played in seattle",
                                        "hypothesis": "the super bowl was played in ohio"})
        assert response.status_code == 200
        results = json.loads(response.data)
        assert "label_probs" in results

    def test_semantic_role_labeling(self):
        response = self.post_json("/predict/semantic-role-labeling",
                                  data={"sentence": "the super bowl was played in seattle"})
        assert response.status_code == 200
        results = json.loads(response.get_data())
        assert "verbs" in results

    def test_caching(self):
        predictor = CountingPredictor()
        data = {"input1": "this is input 1", "input2": 10}
        key = json.dumps(data)

        self.app.predictors["counting"] = predictor

        # call counts should be empty
        assert not predictor.calls

        response = self.post_json("/predict/counting", data=data)
        assert response.status_code == 200
        assert json.loads(response.get_data()) == data

        # call counts should reflect the one call
        assert predictor.calls.get(key) == 1
        assert len(predictor.calls) == 1

        # make a different call
        noyes = {"no": "yes"}
        response = self.post_json("/predict/counting", data=noyes)
        assert response.status_code == 200
        assert json.loads(response.get_data()) == noyes

        # call counts should reflect two calls
        assert predictor.calls[key] == 1
        assert predictor.calls[json.dumps(noyes)] == 1
        assert len(predictor.calls) == 2

        # repeated calls should come from cache and not hit the predictor
        for _ in range(3):
            response = self.post_json("/predict/counting", data=data)
            assert response.status_code == 200
            assert json.loads(response.get_data()) == data

            # these should all be cached, so call counts should not be updated
            assert predictor.calls[key] == 1
            assert predictor.calls[json.dumps(noyes)] == 1
            assert len(predictor.calls) == 2

    def test_disable_caching(self):
        import allennlp.service.server_flask as server
        server.CACHE_SIZE = 0

        predictor = CountingPredictor()
        app = server.make_app(build_dir=self.TEST_DIR)
        app.predictors = {"counting": predictor}
        app.testing = True
        client = app.test_client()

        data = {"input1": "this is input 1", "input2": 10}
        key = json.dumps(data)

        assert not predictor.calls

        for i in range(5):
            response = client.post("/predict/counting",
                                   content_type="application/json",
                                   data=json.dumps(data))
            assert response.status_code == 200
            assert json.loads(response.get_data()) == data

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
        client = app.test_client()

        # Make a prediction, no permalinks.
        data = {"some": "input"}
        response = client.post("/predict/counting", content_type="application/json", data=json.dumps(data))

        assert response.status_code == 200

        # With permalinks not enabled, the result shouldn't contain a slug.
        result = json.loads(response.get_data())
        assert "slug" not in result

        # With permalinks not enabled, a post to the /permadata endpoint should be a 400.
        response = self.client.post("/permadata", data="""{"slug": "someslug"}""")
        assert response.status_code == 400

    def test_permalinks_work(self):
        db = InMemoryDemoDatabase()
        app = make_app(build_dir=self.TEST_DIR, demo_db=db)
        predictor = CountingPredictor()
        app.predictors = {"counting": predictor}
        app.testing = True
        client = app.test_client()

        def post(endpoint: str, data: JsonDict) -> Response:
            return client.post(endpoint, content_type="application/json", data=json.dumps(data))

        data = {"some": "input"}
        response = post("/predict/counting", data=data)

        assert response.status_code == 200
        result = json.loads(response.get_data())
        slug = result.get("slug")
        assert slug is not None

        response = post("/permadata", data={"slug": "not the right slug"})
        assert response.status_code == 400

        response = post("/permadata", data={"slug": slug})
        assert response.status_code == 200
        result2 = json.loads(response.get_data())
        assert set(result2.keys()) == {"modelName", "requestData", "responseData"}
        assert result2["modelName"] == "counting"
        assert result2["requestData"] == data
        assert result2["responseData"] == result
