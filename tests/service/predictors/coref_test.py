# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestCorefPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {"document": "This is a single string document about a test. Sometimes it "
                              "contains coreferent parts."}
        archive = load_archive('tests/fixtures/coref/serialization/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'coreference-resolution')
        result = predictor.predict_json(inputs)

        document = result["document"]
        assert document == ['This', 'is', 'a', 'single', 'string',
                            'document', 'about', 'a', 'test', '.', 'Sometimes',
                            'it', 'contains', 'coreferent', 'parts', '.']

        clusters = result["clusters"]
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, list)
            for mention in cluster:
                # Spans should be integer indices.
                assert isinstance(mention[0], int)
                assert isinstance(mention[1], int)
                # Spans should be inside document.
                assert 0 < mention[0] <= len(document)
                assert 0 < mention[1] <= len(document)
