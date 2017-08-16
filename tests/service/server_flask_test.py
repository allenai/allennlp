# pylint: disable=no-self-use,invalid-name
import json

from allennlp.service.server_flask import make_app
from allennlp.service.predictors import PredictorCollection
from allennlp.common.testing import AllenNlpTestCase

class TestApp(AllenNlpTestCase):

    app = make_app()
    app.testing = True
    app.predictors = PredictorCollection.default()
    client = app.test_client()

    # TODO(joelgrus): this is a fragile test
    def test_list_models(self):
        models = self.client.get("models")
        data = json.loads(models.get_data().decode('utf-8'))
        assert "bidaf" in set(data["models"])

    def test_unknown_model(self):
        prediction = self.client.post("predict/bogus_model",
                                      data="""{"input": "broken"}""",
                                      content_type='application/json')
        assert prediction.status_code == 400
        assert b"unknown model" in prediction.get_data()
