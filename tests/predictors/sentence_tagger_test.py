from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSentenceTaggerPredictor(AllenNlpTestCase):
    def test_predictions_to_labeled_instances(self):
        inputs = {"sentence": "Eric Wallace was an intern at AI2"}

        archive = load_archive(
            self.FIXTURES_ROOT / "simple_tagger" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "sentence_tagger")

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert len(new_instances) > 1
        for new_instance in new_instances:
            assert "tags" in new_instance
            assert len(new_instance["tags"]) == 7  # 7 words in input
