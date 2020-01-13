import numpy as np
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class TestUnidirectionalLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 7)
        self.bidirectional = False
        self.result_keys = {
            "loss",
            "forward_loss",
            "lm_embeddings",
            "noncontextual_token_embeddings",
            "mask",
            "batch_weight",
        }

        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_unidirectional.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )

    def test_unidirectional_language_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=["batch_weight"])

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        print(training_tensors)
        result = self.model(**training_tensors)

        # Unidirectional models should not have backward_loss; bidirectional models should have it.
        assert set(result) == self.result_keys

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == self.expected_embedding_shape

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        if self.bidirectional:
            backward_loss = result["backward_loss"].item()
            np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)
        else:
            np.testing.assert_almost_equal(loss, forward_loss, decimal=3)

    def test_mismatching_contextualizer_unidirectionality_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the contextualizer unidirectionality wrong - it should be
        # False to match the language model.
        params["model"]["contextualizer"]["bidirectional"] = not self.bidirectional
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

    def test_language_model_forward_on_instances(self):
        instances = self.dataset.instances
        predictions = self.model.forward_on_instances(instances)
        assert predictions is not None


class TestUnidirectionalLanguageModelUnsampled(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_unidirectional_unsampled.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )


class TestUnidirectionalLanguageModelTransformer(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 20)

        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_unidirectional_transformer.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )

    def test_unidirectional_language_model_can_train_save_and_load(self):
        # Ignore layer 0 feedforward layer norm parameters, since
        # they are not used.
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore={
                "_contextualizer.feedforward_layer_norm_0.gamma",
                "_contextualizer.feedforward_layer_norm_0.beta",
            },
        )


class TestBidirectionalLanguageModel(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 14)
        self.bidirectional = True
        self.result_keys.add("backward_loss")

        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )


class TestBidirectionalLanguageModelUnsampled(TestBidirectionalLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_unsampled.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )


class TestBidirectionalLanguageModelTransformer(TestBidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 32)

        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_transformer.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )
