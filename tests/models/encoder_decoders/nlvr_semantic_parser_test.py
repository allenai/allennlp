# pylint: disable=no-self-use,protected-access
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase, AllenNlpTestCase
from allennlp.models.encoder_decoders.nlvr_semantic_parser import NlvrDecoderState


class NlvrSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(NlvrSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/nlvr_semantic_parser/experiment.json",
                          "tests/fixtures/data/nlvr/sample_data.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_get_checklist_target(self):
        # Creating a fake all_actions field where actions 0, 2 and 4 are terminal productions.
        all_actions = [{"left": ("", True, {}), "right": ("", False, {})},
                       {"left": ("", True, {}), "right": ("", True, {})},
                       {"left": ("", True, {}), "right": ("", False, {})},
                       {"left": ("", True, {}), "right": ("", True, {})},
                       {"left": ("", True, {}), "right": ("", False, {})}]
        # Of the actions above, those at indices 0 and 4 are on the agenda, and there are padding
        # indices at the end.
        test_agenda = Variable(torch.Tensor([[0], [4], [-1], [-1]]))
        target_checklist, relevant_actions = self.model._get_checklist_target(test_agenda,
                                                                              all_actions)
        assert_almost_equal(target_checklist.data.numpy(), [[1], [0], [1]])
        assert_almost_equal(relevant_actions.data.numpy(), [[0], [2], [4]])


class NlvrDecoderStateTest(AllenNlpTestCase):
    def test_get_checklist_balance(self):
        fake_target = Variable(torch.Tensor([[1], [1], [1], [1], [0]]))
        fake_checklist = Variable(torch.Tensor([[0], [2], [0], [1], [0]]))
        # pylint: disable=protected-access
        balance = NlvrDecoderState._get_checklist_balance(fake_target, fake_checklist)
        assert_almost_equal(balance.data.numpy(), [[1], [0], [1], [0], [0]])
