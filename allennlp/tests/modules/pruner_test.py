# pylint: disable=no-self-use,invalid-name
import numpy
import pytest
import torch

from allennlp.modules import Pruner
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.util import batched_index_select

class TestPruner(AllenNlpTestCase):
    def test_pruner_selects_top_scored_items_and_respects_masking(self):
        # Really simple scorer - sum up the embedding_dim.
        scorer = lambda tensor: tensor.sum(-1).unsqueeze(-1)
        pruner = Pruner(scorer=scorer)

        items = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, :2, :] = 1
        items[1, 2:, :] = 1
        items[2, 2:, :] = 1

        mask = torch.ones([3, 4])
        mask[1, 0] = 0
        mask[1, 3] = 0
        pruned_embeddings, pruned_mask, pruned_indices, pruned_scores = pruner(items, mask, 2)

        # Second element in the batch would have indices 2, 3, but
        # 3 and 0 are masked, so instead it has 1, 2.
        numpy.testing.assert_array_equal(pruned_indices.data.numpy(), numpy.array([[0, 1],
                                                                                   [1, 2],
                                                                                   [2, 3]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.ones([3, 2]))

        # embeddings should be the result of index_selecting the pruned_indices.
        correct_embeddings = batched_index_select(items, pruned_indices)
        numpy.testing.assert_array_equal(correct_embeddings.data.numpy(),
                                         pruned_embeddings.data.numpy())
        # scores should be the sum of the correct embedding elements.
        numpy.testing.assert_array_equal(correct_embeddings.sum(-1).unsqueeze(-1).data.numpy(),
                                         pruned_scores.data.numpy())

    def test_scorer_raises_with_incorrect_scorer_spec(self):
        # Mis-configured scorer - doesn't produce a tensor with 1 as it's final dimension.
        scorer = lambda tensor: tensor.sum(-1)
        pruner = Pruner(scorer=scorer) # type: ignore
        items = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        mask = torch.ones([3, 4])

        with pytest.raises(ValueError):
            _ = pruner(items, mask, 2)

    def test_scorer_works_for_completely_masked_rows(self):
        # Really simple scorer - sum up the embedding_dim.
        scorer = lambda tensor: tensor.sum(-1).unsqueeze(-1)
        pruner = Pruner(scorer=scorer) # type: ignore

        items = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, :2, :] = 1
        items[1, 2:, :] = 1
        items[2, 2:, :] = 1

        mask = torch.ones([3, 4])
        mask[1, 0] = 0
        mask[1, 3] = 0
        mask[2, :] = 0 # fully masked last batch element.

        pruned_embeddings, pruned_mask, pruned_indices, pruned_scores = pruner(items, mask, 2)

        # We can't check the last row here, because it's completely masked.
        # Instead we'll check that the scores for these elements are very small.
        numpy.testing.assert_array_equal(pruned_indices[:2].data.numpy(), numpy.array([[0, 1],
                                                                                       [1, 2]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.array([[1, 1],
                                                                                [1, 1],
                                                                                [0, 0]]))
        # embeddings should be the result of index_selecting the pruned_indices.
        correct_embeddings = batched_index_select(items, pruned_indices)
        numpy.testing.assert_array_equal(correct_embeddings.data.numpy(),
                                         pruned_embeddings.data.numpy())
        # scores should be the sum of the correct embedding elements, with masked elements very
        # small (but not -inf, because that can cause problems).  We'll test these two cases
        # separately.
        correct_scores = correct_embeddings.sum(-1).unsqueeze(-1).data.numpy()
        numpy.testing.assert_array_equal(correct_scores[:2], pruned_scores[:2].data.numpy())
        numpy.testing.assert_array_equal(pruned_scores[2] < -1e15, [[1], [1]])
        numpy.testing.assert_array_equal(pruned_scores[2] == float('-inf'), [[0], [0]])
