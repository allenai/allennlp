# pylint: disable=invalid-name,protected-access
import pytest
import numpy


from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class BiattentiveClassificationNetworkTest(ModelTestCase):
    def setUp(self):
        super(BiattentiveClassificationNetworkTest, self).setUp()
        self.set_up_model('tests/fixtures/biattentive_classification_network/experiment.json',
                          'tests/fixtures/data/sst.txt')

    def test_bcn_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        class_probs = output_dict['class_probabilities'][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1),
                                          numpy.ones(class_probs.shape[0]))

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))
