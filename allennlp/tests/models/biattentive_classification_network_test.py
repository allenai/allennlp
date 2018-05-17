# pylint: disable=invalid-name,protected-access
import pytest


from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class BiattentiveClassificationNetworkMaxoutTest(ModelTestCase):
    def setUp(self):
        super(BiattentiveClassificationNetworkMaxoutTest, self).setUp()
        self.set_up_model('tests/fixtures/biattentive_classification_network/experiment.json',
                          'tests/fixtures/data/sst.txt')

    def test_maxout_bcn_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_feedforward_bcn_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load('tests/fixtures/biattentive_classification_network/feedforward_experiment.json')

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))
