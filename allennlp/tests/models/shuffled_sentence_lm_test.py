# pylint: disable=invalid-name,arguments-differ,abstract-method
import numpy as np
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model

class TestUnidirectionalShuffledSentenceLM(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'experiment_unidirectional.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_shuffled_sentence_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss",
                               "lm_embeddings", "noncontextual_token_embeddings", "mask"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == (2, 8, 7)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()

        np.testing.assert_almost_equal(loss, forward_loss, decimal=3)
        assert result["backward_loss"] is None

    def test_mismatching_contextualizer_unidirectionality_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the contextualizer unidirectionality wrong - it should be
        # False to match the language model.
        params["model"]["contextualizer"]["bidirectional"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))

class TestUnidirectionalShuffledSentenceLMUnsampled(TestUnidirectionalShuffledSentenceLM):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' /
                          'experiment_unidirectional_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

class TestUnidirectionalShuffledSentenceLMTransformer(TestUnidirectionalShuffledSentenceLM):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' /
                          'experiment_unidirectional_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

    # pylint: disable=no-member
    def test_unidirectional_shuffled_sentence_lm_can_train_save_and_load(self):
        # Ignore layer 0 feedforward layer norm parameters, since
        # they are not used.
        self.ensure_model_can_train_save_and_load(
                self.param_file, gradients_to_ignore={
                        "_contextualizer.feedforward_layer_norm_0.gamma",
                        "_contextualizer.feedforward_layer_norm_0.beta"})

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss",
                               "lm_embeddings", "noncontextual_token_embeddings", "mask"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == (2, 8, 20)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()

        np.testing.assert_almost_equal(loss, forward_loss, decimal=3)
        assert result["backward_loss"] is None

class TestBidirectionalShuffledSentenceLM(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

    # pylint: disable=no-member
    def test_bidirectional_shuffled_sentence_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss",
                               "lm_embeddings", "noncontextual_token_embeddings", "mask"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == (2, 8, 14)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()

        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)

    def test_mismatching_contextualizer_bidirectionality_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the contextualizer bidirectionality wrong - it should be
        # true to match the language model.
        params["model"]["contextualizer"]["bidirectional"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get("model"))


class TestBidirectionalShuffledSentenceLMUnsampled(TestBidirectionalShuffledSentenceLM):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'experiment_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

class TestBidirectionalShuffledSentenceLMTransformer(TestBidirectionalShuffledSentenceLM):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'experiment_transformer.jsonnet',
                          self.FIXTURES_ROOT / 'shuffled_sentence_lm' / 'sentences.txt')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss",
                               "lm_embeddings", "noncontextual_token_embeddings", "mask"}

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        # The BidirectionalLanguageModelTransformer uses input size * 2 as the output size unlike
        # a bidirectional LSTM, which uses hidden size * 2.
        assert tuple(embeddings.shape) == (2, 8, 32)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()

        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)
