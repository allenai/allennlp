# pylint: disable=no-self-use,invalid-name
import torch

import pytest

from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder


class TestStackedSelfAttention(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
                self.FIXTURES_ROOT / "encoder_decoder" / "simple_seq2seq" / "multi_gpu_experiment.jsonnet",
                self.FIXTURES_ROOT / "data" / "seq2seq_copy.tsv")

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_works_on_multiple_gpus(self):
        save_dir = self.TEST_DIR / "train_test"
        train_model_from_file(self.param_file, save_dir)

    def test_get_dimension_is_correct(self):
        encoder = StackedSelfAttentionEncoder(input_dim=9,
                                              hidden_dim=12,
                                              projection_dim=6,
                                              feedforward_hidden_dim=5,
                                              num_layers=3,
                                              num_attention_heads=3)
        assert encoder.get_input_dim() == 9
        # hidden_dim + projection_dim
        assert encoder.get_output_dim() == 12

    def test_stacked_self_attention_can_run_foward(self):
        # Correctness checks are elsewhere - this is just stacking
        # blocks which are already well tested, so we just check shapes.
        encoder = StackedSelfAttentionEncoder(input_dim=9,
                                              hidden_dim=12,
                                              projection_dim=9,
                                              feedforward_hidden_dim=5,
                                              num_layers=3,
                                              num_attention_heads=3)
        inputs = torch.randn([3, 5, 9])
        encoder_output = encoder(inputs, None)
        assert list(encoder_output.size()) == [3, 5, 12]
