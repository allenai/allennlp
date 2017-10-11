# pylint: disable=invalid-name
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.nn.util import arrays_to_variables, sequence_cross_entropy_with_logits


class EncoderDecoderWithoutAttentionTest(ModelTestCase):
    def setUp(self):
        super(EncoderDecoderWithoutAttentionTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/experiment.json",
                          "tests/fixtures/data/encoder_decoder.tsv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_loss_is_computed_correctly(self):
        batch_size = 5
        num_decoding_steps = 5
        num_classes = 10
        sample_logits = Variable(torch.randn(batch_size, num_decoding_steps-1, num_classes))
        sample_targets = Variable(torch.from_numpy(numpy.random.randint(0, num_classes,
                                                                        (batch_size, num_decoding_steps))))
        # Mask should be either 0 or 1
        sample_mask = Variable(torch.from_numpy(numpy.random.randint(0, 2,
                                                                     (batch_size, num_decoding_steps))))
        expected_loss = sequence_cross_entropy_with_logits(sample_logits, sample_targets[:, 1:].contiguous(),
                                                           sample_mask[:, 1:].contiguous())
        # pylint: disable=protected-access
        actual_loss = self.model._get_loss(sample_logits, sample_targets, sample_mask)
        assert numpy.equal(expected_loss.data.numpy(), actual_loss.data.numpy())

    def test_decode_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_array_dict())
        output_dict = self.model.forward(**training_arrays)
        decode_output_dict = self.model.decode(output_dict)
        # ``decode`` should have added a ``predicted_tokens`` field to ``output_dict``. Checking if it's there.
        assert "predicted_tokens" in decode_output_dict

class EncoderDecoderWithAttentionTest(ModelTestCase):
    def setUp(self):
        super(EncoderDecoderWithAttentionTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/experiment_with_attention.json",
                          "tests/fixtures/data/encoder_decoder.tsv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
