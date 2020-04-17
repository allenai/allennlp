from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestPredictor(AllenNlpTestCase):
    def test_from_archive_does_not_consume_params(self):
        archive = load_archive(
            self.FIXTURES_ROOT / "simple_tagger" / "serialization" / "model.tar.gz"
        )
        Predictor.from_archive(archive, "sentence-tagger")

        # If it consumes the params, this will raise an exception
        Predictor.from_archive(archive, "sentence-tagger")

    def test_loads_correct_dataset_reader(self):
        # This model has a different dataset reader configuration for train and validation. The
        # parameter that differs is the token indexer's namespace.
        archive = load_archive(
            self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "serialization" / "model.tar.gz"
        )

        predictor = Predictor.from_archive(archive, "sentence-tagger")
        assert predictor._dataset_reader._token_indexers["tokens"].namespace == "test_tokens"

        predictor = Predictor.from_archive(
            archive, "sentence-tagger", dataset_reader_to_load="train"
        )
        assert predictor._dataset_reader._token_indexers["tokens"].namespace == "tokens"

        predictor = Predictor.from_archive(
            archive, "sentence-tagger", dataset_reader_to_load="validation"
        )
        assert predictor._dataset_reader._token_indexers["tokens"].namespace == "test_tokens"

    def test_get_gradients(self):
        inputs = {
            "sentence": "I always write unit tests",
        }

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive)

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        labeled_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        for instance in labeled_instances:
            grads = predictor.get_gradients([instance])[0]
            assert "grad_input_1" in grads
            assert grads["grad_input_1"] is not None
            assert len(grads["grad_input_1"][0]) == 5  # 9 words in hypothesis
