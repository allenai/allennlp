import json

from allennlp.common.testing import ModelTestCase


class ComposedSeq2SeqTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "encoder_decoder" / "composed_seq2seq" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "seq2seq_copy.tsv",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_bidirectional_model_can_train_save_and_load(self):

        param_overrides = json.dumps(
            {
                "model": {
                    "encoder": {"bidirectional": True},
                    "decoder": {"decoder_net": {"decoding_dim": 20, "bidirectional_input": True}},
                }
            }
        )
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )

    def test_no_attention_model_can_train_save_and_load(self):
        param_overrides = json.dumps({"model": {"decoder": {"decoder_net": {"attention": None}}}})
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )

    def test_greedy_model_can_train_save_and_load(self):
        param_overrides = json.dumps({"model": {"decoder": {"beam_size": 1}}})
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )

    def test_decode_runs_correctly(self):
        self.model.eval()
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)
        # `decode` should have added a `predicted_tokens` field to `output_dict`. Checking if it's there.
        assert "predicted_tokens" in decode_output_dict

        # The output of model.decode should still have 'predicted_tokens' after using
        # the beam search. To force the beam search, we just remove `target_tokens`
        # from the input tensors.
        del training_tensors["target_tokens"]
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)
        assert "predicted_tokens" in decode_output_dict
