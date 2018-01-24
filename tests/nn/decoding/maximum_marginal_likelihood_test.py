# pylint: disable=invalid-name, no-self-use,too-many-public-methods
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.decoding import MaximumMarginalLikelihood


class TestMaximumMarginalLikelihood(AllenNlpTestCase):
    def test_create_allowed_transitions(self):
        # pylint: disable=protected-access
        # `1` is our "start symbol" here - the first index is always assumed to be the start symbol
        # and is ignored.  This is due to how the data processing for seq2seq data works.  If that
        # logic changes, we can revisit this.
        targets = torch.autograd.Variable(torch.Tensor([[[1, 2, 3], [1, 4, 5], [1, 2, 7]],
                                                        [[1, 9, 10], [1, 3, 5], [1, 3, 3]]]))
        target_mask = torch.autograd.Variable(torch.Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 0]],
                                                            [[0, 0, 0], [1, 1, 1], [1, 1, 1]]]))
        result = MaximumMarginalLikelihood._create_allowed_transitions(targets, target_mask)
        # There were two instances in this batch.
        assert len(result) == 2

        # The first instance had three valid action sequence prefixes.
        assert len(result[0]) == 3
        assert result[0][()] == set([2, 4])
        assert result[0][(2,)] == set([3])  # note that 7 is not in here; it was masked
        assert result[0][(4,)] == set([5])

        # The second instance had two valid action sequence prefixes.
        assert len(result[1]) == 2
        assert result[1][()] == set([3])  # note that 9 is not in here; it was masked
        assert result[1][(3,)] == set([3, 5])
