# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBiaffineDependencyParser(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "Please could you parse this sentence?",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser'
                               / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

        result = predictor.predict_json(inputs)

        heads = result.get("heads")
        assert heads is not None
        assert isinstance(heads, list)
        assert all(isinstance(x, int) for x in heads)
        head_tags = result.get("head_tags")
        assert head_tags is not None
        assert isinstance(head_tags, list)
        assert all(isinstance(x, int) for x in head_tags)

        predicted_heads = result.get("predicted_heads")
        assert len(predicted_heads) == len(heads) - 1

        predicted_dependencies = result.get("predicted_dependencies")
        assert len(predicted_dependencies) == len(head_tags) - 1
        assert isinstance(predicted_dependencies, list)
        assert all(isinstance(x, str) for x in predicted_dependencies)

        assert result.get("loss") is not None
        assert result.get("arc_loss") is not None
        assert result.get("tag_loss") is not None

    def test_batch_prediction(self):
        inputs = [
                {
                        "sentence": "What kind of test succeeded on its first attempt?",
                },
                {
                        "sentence": "What kind of test succeeded on its first attempt at batch processing?",
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser'
                               / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            sequence_length = sum(result.get("mask")) - 1
            heads = result.get("heads")
            assert heads is not None
            assert isinstance(heads, list)
            assert all(isinstance(x, int) for x in heads)
            head_tags = result.get("head_tags")
            assert head_tags is not None
            assert isinstance(head_tags, list)
            assert all(isinstance(x, int) for x in head_tags)

            predicted_heads = result.get("predicted_heads")
            assert len(predicted_heads) == sequence_length

            predicted_dependencies = result.get("predicted_dependencies")
            assert len(predicted_dependencies) == sequence_length
            assert isinstance(predicted_dependencies, list)
            assert all(isinstance(x, str) for x in predicted_dependencies)
