

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding.chu_liu_edmonds import _find_cycle

class ChuLiuEdmondsTest(AllenNlpTestCase):

    def test_find_cycle(self):
        
        # No cycle
        parents = [0, 2, 3, 0, 3]
        current_nodes = [True for _ in range(5)]
        has_cycle, cycle = _find_cycle(parents, 5, current_nodes)
        assert not has_cycle
        assert not cycle

        # Cycle
        parents = [0, 2, 3, 1, 3]
        has_cycle, cycle = _find_cycle(parents, 5, current_nodes)
        assert has_cycle
        assert cycle == [1, 2, 3]

        # No cycle if ignored nodes are correctly ignored.
        parents = [-1, 0, 1, 4, 3]
        current_nodes = [True for _ in range(5)]
        current_nodes[4] = False
        current_nodes[3] = False
        has_cycle, cycle = _find_cycle(parents, 5, current_nodes)
        assert not has_cycle
        assert cycle == []

        # Cycle, but excluding ignored nodes which form their own cycle.
        parents = [-1, 2, 1, 4, 3]
        current_nodes = [True for _ in range(5)]
        current_nodes[1] = False
        current_nodes[2] = False
        has_cycle, cycle = _find_cycle(parents, 5, current_nodes)
        assert has_cycle
        assert cycle == [3, 4]

        # TODO figure out the weird case where cycles can include ignored nodes.
