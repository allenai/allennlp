# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class TestPredictor(AllenNlpTestCase):
    def test_from_archive_does_not_consume_params(self):
        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        Predictor.from_archive(archive, 'machine-comprehension')

        # If it consumes the params, this will raise an exception
        Predictor.from_archive(archive, 'machine-comprehension')

    def test_get_gradients(self):
        inputs = {
                "premise": "I always write unit tests",
                "hypothesis": "One time I did not write any unit tests"
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')

        labeled_instances = predictor.inputs_to_labeled_instances(inputs)
        for instance in labeled_instances:
            grads = predictor.get_gradients([instance])[0]
            assert 'grad_input_1' in grads
            assert 'grad_input_2' in grads
            assert grads['grad_input_1'] is not None
            assert grads['grad_input_2'] is not None
            assert len(grads['grad_input_1']) == 9  # 9 words in hypothesis
            assert len(grads['grad_input_2']) == 5  # 5 words in premise
