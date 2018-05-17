# pylint: disable=no-self-use,invalid-name,line-too-long
import json
import os

import flask
import flask.testing

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_simple import make_app


def post_json(client: flask.testing.FlaskClient, endpoint: str, data: JsonDict) -> flask.Response:
    return client.post(endpoint,
                       content_type="application/json",
                       data=json.dumps(data))

PAYLOAD = {
        'passage': """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.""",
        'question': """Who stars in the matrix?"""
}


class TestSimpleServer(AllenNlpTestCase):

    def setUp(self):
        super().setUp()

        archive = load_archive('tests/fixtures/bidaf/serialization/model.tar.gz')
        self.bidaf_predictor = Predictor.from_archive(archive, 'machine-comprehension')


    def tearDown(self):
        super().tearDown()
        try:
            os.remove('access.log')
            os.remove('error.log')
        except FileNotFoundError:
            pass

    def test_standard_model(self):
        app = make_app(predictor=self.bidaf_predictor, field_names=['passage', 'question'])
        app.testing = True
        client = app.test_client()

        # First test the HTML
        response = client.get('/')
        data = response.get_data()

        assert b"passage" in data
        assert b"question" in data

        # Now test the backend
        response = post_json(client, '/predict', PAYLOAD)
        data = json.loads(response.get_data())
        assert 'best_span_str' in data
        assert 'span_start_logits' in data

    def test_sanitizer(self):
        def sanitize(result: JsonDict) -> JsonDict:
            return {key: value for key, value in result.items()
                    if key.startswith("best_span")}

        app = make_app(predictor=self.bidaf_predictor, field_names=['passage', 'question'], sanitizer=sanitize)
        app.testing = True
        client = app.test_client()

        response = post_json(client, '/predict', PAYLOAD)
        data = json.loads(response.get_data())
        assert 'best_span_str' in data
        assert 'span_start_logits' not in data

    def test_static_dir(self):
        html = """<html><body>THIS IS A STATIC SITE</body></html>"""
        jpg = """something about a jpg"""

        with open(os.path.join(self.TEST_DIR, 'index.html'), 'w') as f:
            f.write(html)

        with open(os.path.join(self.TEST_DIR, 'jpg.txt'), 'w') as f:
            f.write(jpg)

        app = make_app(predictor=self.bidaf_predictor, static_dir=self.TEST_DIR)
        app.testing = True
        client = app.test_client()

        response = client.get('/')
        data = response.get_data().decode('utf-8')
        assert data == html

        response = client.get('jpg.txt')
        data = response.get_data().decode('utf-8')
        assert data == jpg
