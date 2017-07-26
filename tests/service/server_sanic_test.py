# pylint: disable=no-self-use,invalid-name
import json
import os

from allennlp.service.server_sanic import app
from allennlp.testing.test_case import AllenNlpTestCase


class TestApp(AllenNlpTestCase):

    def tearDown(self):
        super(TestApp, self).tearDown()
        try:
            os.remove('access.log')
            os.remove('error.log')
        except FileNotFoundError:
            pass

    def test_list_models(self):
        app.testing = True
        client = app.test_client

        _, response = client.get("/models")
        data = json.loads(response.text)
        assert "reverser" in set(data["models"])

    def test_unknown_model(self):
        app.testing = True
        client = app.test_client
        _, response = client.post("/predict/bogus_model",
                                  json={"input": "broken"})
        assert response.status == 400
        assert "unknown model" in response.text and "bogus_model" in response.text

    def test_known_model(self):
        app.testing = True
        client = app.test_client
        _, response = client.post("/predict/reverser",
                                  json={"input": "not broken"})
        data = json.loads(response.text)
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "reverser"
        assert data["input"] == "not broken"
        assert data["output"] == "nekorb ton"
