# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server import app
from allennlp.testing.test_case import AllenNlpTestCase



class TestApp(AllenNlpTestCase):

    def test_list_models(self):
        app.testing = True
        client = app.test_client()

        models = client.get("models")
        data = json.loads(models.get_data().decode('utf-8'))
        assert set(data["models"]) == {"uppercase", "lowercase", "reverse"}

    def test_unknown_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/bogus_model",
                                 data="""{"input": "broken"}""",
                                 content_type='application/json')
        assert prediction.status_code == 400
        assert b"unknown model" in prediction.get_data()

    def test_known_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/reverse",
                                 data="""{"input": "not broken"}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "reverse"
        assert data["input"] == "not broken"
        assert data["output"] == "nekorb ton"
