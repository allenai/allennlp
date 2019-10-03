import numpy as np

from allennlp.common.testing import ModelTestCase


class TestBidirectionalLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()

        self.expected_embedding_shape = (2, 8, 14)

        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_bidirectional.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )

    def test_bidirectional_lm_can_train_save_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=["batch_weight"])

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {
            "loss",
            "forward_loss",
            "backward_loss",
            "lm_embeddings",
            "noncontextual_token_embeddings",
            "mask",
            "batch_weight",
        }

        # The model should preserve the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == self.expected_embedding_shape

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()
        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)


class TestBidirectionalLanguageModelUnsampled(TestBidirectionalLanguageModel):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "language_model" / "experiment_bidirectional_unsampled.jsonnet",
            self.FIXTURES_ROOT / "language_model" / "sentences.txt",
        )
