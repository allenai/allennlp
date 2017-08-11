# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server_sanic import make_app
from allennlp.service.servable import ServableCollection
from allennlp.common.testing import AllenNlpTestCase


class TestApp(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        app = make_app()
        app.servables = ServableCollection.default()
        app.testing = True
        self.client = app.test_client

    def test_list_models(self):
        _, response = self.client.get("/models")
        data = json.loads(response.text)
        assert "bidaf" in set(data["models"])

    def test_unknown_model(self):
        _, response = self.client.post("/predict/bogus_model",
                                       json={"input": "broken"})
        assert response.status == 400
        assert "unknown model" in response.text and "bogus_model" in response.text

    def test_bidaf(self):
        _, response = self.client.post("/predict/bidaf",
                                       json={"passage": "the super bowl was played in seattle",
                                             "question": "where was the super bowl played?"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "best_span" in results

    def test_snli(self):
        _, response = self.client.post("/predict/snli",
                                       json={"premise": "the super bowl was played in seattle",
                                             "hypothesis": "the super bowl was played in ohio"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "label_probs" in results

    def test_srl(self):
        _, response = self.client.post("/predict/srl",
                                       json={"sentence": "the super bowl was played in seattle"})
        assert response.status == 200
        results = json.loads(response.text)
        assert "verbs" in results
