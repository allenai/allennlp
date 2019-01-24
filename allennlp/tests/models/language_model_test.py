# pylint: disable=invalid-name,arguments-differ,abstract-method
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

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_unidirectional.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_language_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=["batch_weight"])

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings",
                               "noncontextual_token_embeddings", "mask", "batch_weight"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == self.expected_embedding_shape

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        if self.bidirectional:
            backward_loss = result["backward_loss"].item()
            np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2,
                                           decimal=3)
        else:
            np.testing.assert_almost_equal(loss, forward_loss, decimal=3)
            assert result["backward_loss"] is None

    def test_mismatching_contextualizer_unidirectionality_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the contextualizer unidirectionality wrong - it should be
        # False to match the language model.
        params["model"]["contextualizer"]["bidirectional"] = (not self.bidirectional)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

class TestUnidirectionalLanguageModelUnsampled(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' /
                          'experiment_unidirectional_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestUnidirectionalLanguageModelTransformer(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 20)

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' /
                          'experiment_unidirectional_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_language_model_can_train_save_and_load(self):
        # Ignore layer 0 feedforward layer norm parameters, since
        # they are not used.
        self.ensure_model_can_train_save_and_load(
                self.param_file, gradients_to_ignore={
                        "_contextualizer.feedforward_layer_norm_0.gamma",
                        "_contextualizer.feedforward_layer_norm_0.beta"})

class TestUnidirectionalContiguousLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 6, 7)
        self.bidirectional = False

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' /
                          'experiment_unidirectional_contiguous.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_language_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        # Note: The dataset itself generates an extra singleton dimension in the
        # first dimension. This dimension is squeezed out in the
        # LanguageModelingIterator, but we need to do it manually here.
        for token_level in training_tensors.get("inputs", {}):
            training_tensors["inputs"][token_level] = training_tensors["inputs"][token_level].squeeze(0)
        for token_level in training_tensors.get("forward_targets", {}):
            training_tensors["forward_targets"][token_level] = training_tensors[
                    "forward_targets"][token_level].squeeze(0)
        for token_level in training_tensors.get("backward_targets", {}):
            training_tensors["backward_targets"][token_level] = training_tensors[
                    "backward_targets"][token_level].squeeze(0)

        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings",
                               "noncontextual_token_embeddings", "mask", "batch_weight"}

        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == self.expected_embedding_shape

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        if self.bidirectional:
            backward_loss = result["backward_loss"].item()
            np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2,
                                           decimal=3)
        else:
            np.testing.assert_almost_equal(loss, forward_loss, decimal=3)
            assert result["backward_loss"] is None

    def test_mismatching_contextualizer_unidirectionality_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the contextualizer unidirectionality wrong - it should be
        # False to match the language model.
        params["model"]["contextualizer"]["bidirectional"] = (not self.bidirectional)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

class TestUnidirectionalContiguousLanguageModelUnsampled(TestUnidirectionalContiguousLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' /
                          'experiment_unidirectional_contiguous_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestUnidirectionalContiguousLanguageModelTransformer(TestUnidirectionalContiguousLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 6, 20)

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' /
                          'experiment_unidirectional_contiguous_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_language_model_can_train_save_and_load(self):
        # Ignore layer 0 feedforward layer norm parameters, since
        # they are not used.
        self.ensure_model_can_train_save_and_load(
                self.param_file, gradients_to_ignore={
                        "_contextualizer.feedforward_layer_norm_0.gamma",
                        "_contextualizer.feedforward_layer_norm_0.beta"})


class TestBidirectionalLanguageModel(TestUnidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 14)
        self.bidirectional = True

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestBidirectionalLanguageModelUnsampled(TestBidirectionalLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestBidirectionalLanguageModelTransformer(TestBidirectionalLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 32)

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestBidirectionalContiguousLanguageModel(TestUnidirectionalContiguousLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 5, 14)
        self.bidirectional = True

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_contiguous.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestBidirectionalContiguousLanguageModelUnsampled(TestBidirectionalContiguousLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_contiguous_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')

class TestBidirectionalContiguousLanguageModelTransformer(TestBidirectionalContiguousLanguageModel):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 5, 32)

        self.set_up_model(self.FIXTURES_ROOT / 'language_model' / 'experiment_contiguous_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'language_model' / 'sentences.txt')
