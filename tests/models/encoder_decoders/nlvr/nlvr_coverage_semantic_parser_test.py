# pylint: disable=no-self-use,protected-access,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.iterators import EpochTrackingBucketIterator
from allennlp.models import Model


class NlvrCoverageSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(NlvrCoverageSemanticParserTest, self).setUp()
        self.set_up_model("tests/fixtures/encoder_decoder/nlvr_coverage_semantic_parser/ungrouped_experiment.json",
                          "tests/fixtures/data/nlvr/sample_ungrouped_data.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_grouped_model_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load("tests/fixtures/encoder_decoder/nlvr_coverage_semantic_parser/experiment.json")

    def test_get_checklist_info(self):
        # Creating a fake all_actions field where actions 0, 2 and 4 are terminal productions.
        all_actions = [{"left": ("", True, {}), "right": ("", False, {})},
                       {"left": ("", True, {}), "right": ("", True, {})},
                       {"left": ("", True, {}), "right": ("", False, {})},
                       {"left": ("", True, {}), "right": ("", True, {})},
                       {"left": ("", True, {}), "right": ("", False, {})}]
        # Of the actions above, those at indices 0 and 4 are on the agenda, and there are padding
        # indices at the end.
        test_agenda = Variable(torch.Tensor([[0], [4], [-1], [-1]]))
        checklist_info = self.model._get_checklist_info(test_agenda, all_actions)
        target_checklist, terminal_actions, checklist_mask = checklist_info
        assert_almost_equal(target_checklist.data.numpy(), [[1], [0], [1]])
        assert_almost_equal(terminal_actions.data.numpy(), [[0], [2], [4]])
        assert_almost_equal(checklist_mask.data.numpy(), [[1], [1], [1]])

    def test_forward_with_epoch_num_changes_cost_weight(self):
        # Redefining model. We do not want this to change the state of ``self.model``.
        params = Params.from_file(self.param_file)
        model = Model.from_params(self.vocab, params['model'])
        # Initial cost weight, before forward is called.
        assert model._checklist_cost_weight == 0.8
        iterator = EpochTrackingBucketIterator(sorting_keys=[['sentence', 'num_tokens']])
        cost_weights = []
        for epoch_data in iterator(self.dataset, num_epochs=4):
            model.forward(**epoch_data)
            cost_weights.append(model._checklist_cost_weight)
        # The config file has ``wait_num_epochs`` set to 0, so the model starts decreasing the cost
        # weight at epoch 0 itself.
        assert_almost_equal(cost_weights, [0.72, 0.648, 0.5832, 0.52488])
