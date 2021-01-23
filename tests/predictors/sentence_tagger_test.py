from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSentenceTaggerPredictor(AllenNlpTestCase):
    def test_predictions_to_labeled_instances(self):
        inputs = {"sentence": "cats are animals."}

        archive = load_archive(
            self.FIXTURES_ROOT / "simple_tagger" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "sentence_tagger")

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        assert outputs["tags"] == ["N", "V", "N", "N"]
