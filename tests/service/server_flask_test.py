# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server_flask import make_app
from allennlp.service.servable import ServableCollection
from allennlp.common.testing import AllenNlpTestCase

# Add default models to app
app = make_app()
app.servables = ServableCollection.default()

class TestApp(AllenNlpTestCase):

    # TODO(joelgrus): this is a fragile test
    def test_list_models(self):
        app.testing = True
        client = app.test_client()

        models = client.get("models")
        data = json.loads(models.get_data().decode('utf-8'))
        assert "bidaf" in set(data["models"])

    def test_unknown_model(self):
        app.testing = True
        client = app.test_client()
        prediction = client.post("predict/bogus_model",
                                 data="""{"input": "broken"}""",
                                 content_type='application/json')
        assert prediction.status_code == 400
        assert b"unknown model" in prediction.get_data()
