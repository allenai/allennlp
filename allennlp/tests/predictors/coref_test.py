# pylint: disable=no-self-use,invalid-name
import spacy

from allennlp.common.testing import AllenNlpTestCase

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.predictors import CorefPredictor


class TestCorefPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {"document": "This is a single string document about a test. Sometimes it "
                              "contains coreferent parts."}
        archive = load_archive(self.FIXTURES_ROOT / 'coref' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'coreference-resolution')

        result = predictor.predict_json(inputs)
        self.assert_predict_result(result)

        document = ['This', 'is', 'a', 'single', 'string',
                    'document', 'about', 'a', 'test', '.', 'Sometimes',
                    'it', 'contains', 'coreferent', 'parts', '.']

        result_doc_words = predictor.predict_tokenized(document)
        self.assert_predict_result(result_doc_words)

    @staticmethod
    def assert_predict_result(result):
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

    def test_coref_resolved(self):

        """Tests I/O of coref_resolved method"""

        document = "This is a test sentence."
        archive = load_archive(self.FIXTURES_ROOT / 'coref' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'coreference-resolution')
        result = predictor.coref_resolved(document)
        assert isinstance(result, str)

    def test_replace_corefs(self):

        """Tests core coref replacement logic"""

        nlp = spacy.load("en_core_web_sm")

        inputs = [
                "This is a sentence with no coreferences.", # No coreferences
                "Julie wants to buy fruit. That is what she loves.", # Single coreference / personal
                "Charlie wants to buy a game, so he can play it with friends.", # Multiple coreferences / personal
                "The woman reading a newspaper sat on the bench with her dog.", # Phrasal mention / possessive
                "Canada stimulated the country's economy." # Phrasal coreference / possessive
                ]

        expected_clusters = [
                [],
                [[[0, 0], [9, 9]]],
                [[[0, 0], [8, 8]], [[4, 5], [11, 11]]],
                [[[0, 4], [10, 10]]],
                [[[0, 0], [2, 4]]]
                ]

        expected_outputs = [
                "This is a sentence with no coreferences.",
                "Julie wants to buy fruit. That is what Julie loves.",
                "Charlie wants to buy a game, so Charlie can play a game with friends.",
                "The woman reading a newspaper sat on the bench with The woman reading a newspaper's dog.",
                "Canada stimulated Canada's economy."
                ]

        for i, text in enumerate(inputs):
            clusters = expected_clusters[i]

            if not clusters:
                assert text == inputs[i]
                continue

            doc = nlp(text)
            output = CorefPredictor.replace_corefs(doc, clusters)
            assert output == expected_outputs[i]
    def test_predictions_to_labeled_instances(self):
        inputs = {"document": "This is a single string document about a test. Sometimes it "
                              "contains coreferent parts."}
        archive = load_archive(self.FIXTURES_ROOT / 'coref' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'coreference-resolution')

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert new_instances is not None

        for new_instance in new_instances:
            assert 'span_labels' in new_instance
            assert len(new_instance['span_labels']) == 60 # 7 words in input
            true_top_spans = set(tuple(span) for span in outputs['top_spans'])
            pred_clust_spans = set()
            for i, span in enumerate(outputs['top_spans']):
                if new_instance['span_labels'][i]:
                    pred_clust_spans.add(tuple(span))
            assert true_top_spans == pred_clust_spans

x = TestCorefPredictor()
x.test_predictions_to_labeled_instances()
