# pylint: disable=no-self-use,invalid-name,protected-access
from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBidafPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        result = predictor.predict_json(inputs)

        best_span = result.get("best_span")
        assert best_span is not None
        assert isinstance(best_span, list)
        assert len(best_span) == 2
        assert all(isinstance(x, int) for x in best_span)
        assert best_span[0] <= best_span[1]

        best_span_str = result.get("best_span_str")
        assert isinstance(best_span_str, str)
        assert best_span_str != ""

        for probs_key in ("span_start_probs", "span_end_probs"):
            probs = result.get(probs_key)
            assert probs is not None
            assert all(isinstance(x, float) for x in probs)
            assert sum(probs) == approx(1.0)

    def test_batch_prediction(self):
        inputs = [
                {
                        "question": "What kind of test succeeded on its first attempt?",
                        "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
                },
                {
                        "question": "What kind of test succeeded on its first attempt at batch processing?",
                        "passage": "One time I was writing a unit test, and it always failed!"
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            best_span = result.get("best_span")
            best_span_str = result.get("best_span_str")
            start_probs = result.get("span_start_probs")
            end_probs = result.get("span_end_probs")
            assert best_span is not None
            assert isinstance(best_span, list)
            assert len(best_span) == 2
            assert all(isinstance(x, int) for x in best_span)
            assert best_span[0] <= best_span[1]

            assert isinstance(best_span_str, str)
            assert best_span_str != ""

            for probs in (start_probs, end_probs):
                assert probs is not None
                assert all(isinstance(x, float) for x in probs)
                assert sum(probs) == approx(1.0)

    def test_model_internals(self):
        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        # Context manager to capture model internals
        with predictor.capture_model_internals() as internals:
            predictor.predict_json(inputs)

        assert internals is not None
        assert len(internals) == 25

        linear_50_1 = internals[23]
        assert "Linear(in_features=50, out_features=1, bias=True)" in linear_50_1["name"]
        assert len(linear_50_1['output']) == 17
        assert all(len(a) == 1 for a in linear_50_1['output'])

        # hooks should be gone
        for module in predictor._model.modules():
            assert not module._forward_hooks

    def test_predictions_to_labeled_instances(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert 'span_start' in new_instances[0].fields
        assert 'span_end' in new_instances[0].fields
        assert new_instances[0].fields['span_start'] is not None
        assert new_instances[0].fields['span_end'] is not None
        assert len(new_instances) == 1

    def test_predictions_to_labeled_instances_with_naqanet(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing 2 unit tests, and 1 succeeded on the first attempt."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'naqanet' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')
        predictor._dataset_reader.skip_when_all_empty = False

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert 'number_indices' in new_instances[0].fields
        assert 'answer_as_passage_spans' in new_instances[0].fields
        assert 'answer_as_question_spans' in new_instances[0].fields
        assert 'answer_as_add_sub_expressions' in new_instances[0].fields
        assert 'answer_as_counts' in new_instances[0].fields
        assert 'metadata' in new_instances[0].fields
        assert len(new_instances) == 1

        outputs['answer']['answer_type'] = 'count'
        outputs['answer']['count'] = 2
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert new_instances[0]['answer_as_counts'][0].label == 2

        outputs['answer']['answer_type'] = 'passage_span'
        outputs['answer']['spans'] = [[0, 8]]  # character offsets
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert new_instances[0]['answer_as_passage_spans'][0] == (0, 1)  # token indices

        outputs['answer']['answer_type'] = 'arithmetic'
        outputs['answer']['numbers'] = [{'sign': 2}, {'sign': 0}]
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert new_instances[0]['answer_as_add_sub_expressions'][0].labels == [2, 0, 0]

        outputs['answer']['answer_type'] = 'question_span'
        outputs['answer']['spans'] = [[0, 9]]  # character offsets
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert new_instances[0]['answer_as_question_spans'][0] == (0, 1)  # token indices
