from flaky import flaky
import numpy

from allennlp.common.testing import ModelTestCase


class BiasMitigatorApplicatorTest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            self.FIXTURES_ROOT / "fairness" / "experiment.jsonnet",
            self.FIXTURES_ROOT / "fairness" / "snli_train.jsonl",
        )

    def test_bias_mitigator_applicator_can_train_save_and_load(self):
        # BertModel pooler output is discarded so grads not computed
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            gradients_to_ignore=set(
                [
                    "base_model._text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.weight",
                    "base_model._text_field_embedder.token_embedder_tokens.transformer_model.pooler.dense.bias",
                ]
            ),
        )

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        assert "label" in output_dict.keys()
        probs = output_dict["probs"][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(probs, -1), numpy.array([1]))

    def test_forward_on_instances_ignores_loss_key_when_batched(self):
        batch_outputs = self.model.forward_on_instances(self.dataset.instances)
        for output in batch_outputs:
            assert "loss" not in output.keys()

        # It should be in the single batch case, because we special case it.
        single_output = self.model.forward_on_instance(self.dataset.instances[0])
        assert "loss" in single_output.keys()
