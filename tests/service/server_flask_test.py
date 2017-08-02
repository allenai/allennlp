# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server_flask import app
from allennlp.testing.test_case import AllenNlpTestCase


class TestApp(AllenNlpTestCase):

    # TODO(joelgrus): this is a fragile test
    def test_list_models(self):
        app.testing = True
        client = app.test_client()

        models = client.get("models")
        data = json.loads(models.get_data().decode('utf-8'))
        assert "reverser" in set(data["models"])

    def test_unknown_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/bogus_model",
                                 data="""{"input": "broken"}""",
                                 content_type='application/json')
        assert prediction.status_code == 400
        assert b"unknown model" in prediction.get_data()

    def test_reverse_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/reverser",
                                 data="""{"input": "not broken"}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "reverser"
        assert data["input"] == "not broken"
        assert data["output"] == "nekorb ton"

    def test_upper_lowercase_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/lowercaser",
                                 data="""{"input": "UPPERcase"}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "lowercaser"
        assert data["input"] == "UPPERcase"
        assert data["output"] == "uppercase"

        prediction = client.post("predict/uppercaser",
                                 data="""{"input": "lowerCASE"}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "uppercaser"
        assert data["input"] == "lowerCASE"
        assert data["output"] == "LOWERCASE"

    def test_pytorch_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/matrix_multiplier",
                                 data="""{"input": 1}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        assert set(data.keys()) == {"input", "model_name", "output"}
        assert data["model_name"] == "matrix_multiplier"
        assert data["input"] == 1
        assert isinstance(data["output"], list)
        assert isinstance(data["output"][0], list)

    def test_simple_tagger_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/simple_tagger",
                                 data="""{"input": "the cat is here"}""",
                                 content_type='application/json')
        data = json.loads(prediction.get_data().decode('utf-8'))
        # TODO(joelgrus): write a better test
        assert set(data.keys()) == {'model_name', 'input', 'output', 'tokens', 'possible_tags'}
