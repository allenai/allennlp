from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestMaskedLanguageModelPredictor(AllenNlpTestCase):
    def test_predictions_to_labeled_instances(self):
        inputs = {"sentence": "Eric [MASK] was an intern at [MASK]"}

        archive = load_archive(
            self.FIXTURES_ROOT / "masked_language_model" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "masked_language_model")

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert len(new_instances) == 1
        assert "target_ids" in new_instances[0]
        assert len(new_instances[0]["target_ids"].tokens) == 2  # should have added two words
