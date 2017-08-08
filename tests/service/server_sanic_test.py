# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server_sanic import make_app
from allennlp.service.servable import ServableCollection
from allennlp.common.testing import AllenNlpTestCase

app = make_app()
# Add default models to app
app.servables = ServableCollection.default()

class TestApp(AllenNlpTestCase):

    def test_list_models(self):
        app.testing = True
        client = app.test_client

        _, response = client.get("/models")
        data = json.loads(response.text)
        assert "bidaf" in set(data["models"])

    def test_unknown_model(self):
        app.testing = True
        client = app.test_client
        _, response = client.post("/predict/bogus_model",
                                  json={"input": "broken"})
        assert response.status == 400
        assert "unknown model" in response.text and "bogus_model" in response.text
