from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestPredictor(AllenNlpTestCase):
    def test_from_archive_does_not_consume_params(self):
        archive = load_archive(self.FIXTURES_ROOT / "bidaf" / "serialization" / "model.tar.gz")
        Predictor.from_archive(archive, "machine-comprehension")

        # If it consumes the params, this will raise an exception
        Predictor.from_archive(archive, "machine-comprehension")

    def test_loads_correct_dataset_reader(self):
        # The NAQANET archive has both a training and validation ``DatasetReader``
        # with different values for ``passage_length_limit`` (``1000`` for validation
        # and ``400`` for training).
        archive = load_archive(self.FIXTURES_ROOT / "naqanet" / "serialization" / "model.tar.gz")

        predictor = Predictor.from_archive(archive, "machine-comprehension")
        assert predictor._dataset_reader.passage_length_limit == 1000

        predictor = Predictor.from_archive(
            archive, "machine-comprehension", dataset_reader_to_load="train"
        )
        assert predictor._dataset_reader.passage_length_limit == 400

        predictor = Predictor.from_archive(
            archive, "machine-comprehension", dataset_reader_to_load="validation"
        )
        assert predictor._dataset_reader.passage_length_limit == 1000

    def test_get_gradients(self):
        inputs = {
            "premise": "I always write unit tests",
            "hypothesis": "One time I did not write any unit tests",
        }

        archive = load_archive(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "textual-entailment")

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        labeled_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        for instance in labeled_instances:
            grads = predictor.get_gradients([instance])[0]
            assert "grad_input_1" in grads
            assert "grad_input_2" in grads
            assert grads["grad_input_1"] is not None
            assert grads["grad_input_2"] is not None
            assert len(grads["grad_input_1"][0]) == 9  # 9 words in hypothesis
            assert len(grads["grad_input_2"][0]) == 5  # 5 words in premise
