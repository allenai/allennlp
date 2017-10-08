# pylint: disable=invalid-name
from allennlp.common.testing import ModelTestCase


class EncoderDecoderTest(ModelTestCase):
    def setUp(self):
        super(EncoderDecoderTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/experiment.json",
                          "tests/fixtures/data/encoder_decoder.tsv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
