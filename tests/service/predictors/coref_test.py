# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestCorefPredictor(TestCase):

    def test_uses_named_inputs(self):

        inputs = {"document": "This is a single string document about a test. Sometimes it "
                              "contains coreferent parts."}
        archive = load_archive('tests/fixtures/coref/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'coref')
        result = predictor.predict_json(inputs)
