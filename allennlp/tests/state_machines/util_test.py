# pylint: disable=invalid-name,no-self-use,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.state_machines import util


class TestStateMachinesUtil(AllenNlpTestCase):
    def test_create_allowed_transitions(self):
        targets = torch.Tensor([[[2, 3, 4], [1, 3, 4], [1, 2, 4]], [[3, 4, 0], [2, 3, 4], [0, 0, 0]]])
        target_mask = torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 0], [1, 1, 1], [0, 0, 0]]])
        prefix_tree = util.construct_prefix_tree(targets, target_mask)

        # There were two instances in this batch.
        assert len(prefix_tree) == 2

        # The first instance had six valid action sequence prefixes.
        assert len(prefix_tree[0]) == 6
        assert prefix_tree[0][()] == {1, 2}
        assert prefix_tree[0][(1,)] == {2, 3}
        assert prefix_tree[0][(1, 2)] == {4}
        assert prefix_tree[0][(1, 3)] == {4}
        assert prefix_tree[0][(2,)] == {3}
        assert prefix_tree[0][(2, 3)] == {4}

        # The second instance had four valid action sequence prefixes.
        assert len(prefix_tree[1]) == 4
        assert prefix_tree[1][()] == {2, 3}
        assert prefix_tree[1][(2,)] == {3}
        assert prefix_tree[1][(2, 3)] == {4}
        assert prefix_tree[1][(3,)] == {4}
