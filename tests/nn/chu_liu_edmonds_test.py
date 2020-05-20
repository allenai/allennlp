import numpy
import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.nn.chu_liu_edmonds import _find_cycle, decode_mst


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

    def test_mst(self):
        # First, test some random cases as sanity checks.
        # No label case
        energy = numpy.random.rand(5, 5)
        heads, types = decode_mst(energy, 5, has_labels=False)
        assert not _find_cycle(heads, 5, [True] * 5)[0]

        # Labeled case
        energy = numpy.random.rand(3, 5, 5)
        heads, types = decode_mst(energy, 5)

        assert not _find_cycle(heads, 5, [True] * 5)[0]
        label_id_matrix = energy.argmax(axis=0)

        # Check that the labels correspond to the
        # argmax of the labels for the arcs.
        for child, parent in enumerate(heads):
            # The first index corresponds to the symbolic
            # head token, which won't necessarily have an
            # argmax type.
            if child == 0:
                continue
            assert types[child] == label_id_matrix[parent, child]

        # Check wrong dimensions throw errors
        with pytest.raises(ConfigurationError):
            energy = numpy.random.rand(5, 5)
            decode_mst(energy, 5, has_labels=True)

        with pytest.raises(ConfigurationError):
            energy = numpy.random.rand(3, 5, 5)
            decode_mst(energy, 5, has_labels=False)

    def test_mst_finds_maximum_spanning_tree(self):
        energy = torch.arange(1, 10).view(1, 3, 3)
        heads, _ = decode_mst(energy.numpy(), 3)
        assert list(heads) == [-1, 2, 0]
