# pylint: disable=invalid-name
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.nn.util import arrays_to_variables, sequence_cross_entropy_with_logits


class WikiTablesSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(WikiTablesSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/wikitables_semantic_parser/experiment.json",
                          "tests/fixtures/data/seq2seq_copy.tsv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
