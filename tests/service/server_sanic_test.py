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
