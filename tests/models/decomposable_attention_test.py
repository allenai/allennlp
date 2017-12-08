# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pytest
import numpy
from numpy.testing import assert_almost_equal

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.models import DecomposableAttention, Model


class TestDecomposableAttention(ModelTestCase):
    def setUp(self):
        super(TestDecomposableAttention, self).setUp()
        self.set_up_model('tests/fixtures/decomposable_attention/experiment.json',
                          'tests/fixtures/data/snli.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert_almost_equal(numpy.sum(output_dict["label_probs"][0].data.numpy(), -1), 1, decimal=6)

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_model_load(self):
        params = Params.from_file('tests/fixtures/decomposable_attention/experiment.json')
        model = Model.load(params, serialization_dir='tests/fixtures/decomposable_attention/serialization')

        assert isinstance(model, DecomposableAttention)

    def test_mismatched_dimensions_raise_configuration_errors(self):
        params = Params.from_file(self.param_file)
        # Make the input_dim to the first feedforward_layer wrong - it should be 2.
        params["model"]["attend_feedforward"]["input_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

        params = Params.from_file(self.param_file)
        # Make the projection output_dim of the last layer wrong - it should be
        # 3, equal to the number of classes.
        params["model"]["aggregate_feedforward"]["output_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))
