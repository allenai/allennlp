# pylint: disable=invalid-name,arguments-differ,abstract-method
import numpy as np

from allennlp.common.testing import ModelTestCase

class TestBidirectionalLM(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'bidirectional_lm' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'bidirectional_lm' / 'sentences.txt')

    # pylint: disable=no-member
    def test_bidirectional_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings", "mask"}

        # The model should have removed the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == (2, 6, 14)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()

        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)


class TestBidirectionalLMUnsampled(TestBidirectionalLM):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'bidirectional_lm' / 'experiment_unsampled.jsonnet',
                          self.FIXTURES_ROOT / 'bidirectional_lm' / 'sentences.txt')
