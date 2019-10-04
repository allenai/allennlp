from copy import deepcopy
import pytest


from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class BiattentiveClassificationNetworkTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "biattentive_classification_network" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "sst.txt",
        )

    def test_maxout_bcn_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_feedforward_bcn_can_train_save_and_load(self):

        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT
            / "biattentive_classification_network"
            / "feedforward_experiment.json"
        )

    def test_input_and_output_elmo_bcn_can_train_save_and_load(self):

        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / "biattentive_classification_network" / "elmo_experiment.json"
        )

    def test_output_only_elmo_bcn_can_train_save_and_load(self):

        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT
            / "biattentive_classification_network"
            / "output_only_elmo_experiment.json"
        )

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

    def test_no_elmo_but_set_flags_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # There is no elmo specified in self.param_file, but set
        # use_input_elmo and use_integrator_output_elmo to True.
        # use_input_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params["model"]["use_input_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

        # use_integrator_output_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params["model"]["use_input_elmo"] = False
        tmp_params["model"]["use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

        # both use_input_elmo and use_integrator_output_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params["model"]["use_input_elmo"] = True
        tmp_params["model"]["use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

    def test_elmo_but_no_set_flags_throws_configuration_error(self):

        params = Params.from_file(
            self.FIXTURES_ROOT / "biattentive_classification_network" / "elmo_experiment.json"
        )
        # Elmo is specified in the model, but set both flags to false.
        params["model"]["use_input_elmo"] = False
        params["model"]["use_integrator_output_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

    def test_elmo_num_repr_set_flags_mismatch_throws_configuration_error(self):

        params = Params.from_file(
            self.FIXTURES_ROOT / "biattentive_classification_network" / "elmo_experiment.json"
        )
        # Elmo is specified in the model, with num_output_representations=2. Set
        # only one flag to true.
        tmp_params = deepcopy(params)
        tmp_params["model"]["use_input_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

        tmp_params = deepcopy(params)
        tmp_params["model"]["use_input_elmo"] = True
        tmp_params["model"]["use_integrator_output_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

        # set num_output_representations to 1, and set both flags to True.
        tmp_params = deepcopy(params)
        tmp_params["model"]["elmo"]["num_output_representations"] = 1
        tmp_params["model"]["use_input_elmo"] = True
        tmp_params["model"]["use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get("model"))

    def test_no_elmo_tokenizer_throws_configuration_error(self):
        with pytest.raises(ConfigurationError):

            self.ensure_model_can_train_save_and_load(
                self.FIXTURES_ROOT
                / "biattentive_classification_network"
                / "broken_experiments"
                / "no_elmo_tokenizer_for_elmo.json"
            )

    def test_elmo_in_text_field_embedder_throws_configuration_error(self):
        with pytest.raises(ConfigurationError):

            self.ensure_model_can_train_save_and_load(
                self.FIXTURES_ROOT
                / "biattentive_classification_network"
                / "broken_experiments"
                / "elmo_in_text_field_embedder.json"
            )
