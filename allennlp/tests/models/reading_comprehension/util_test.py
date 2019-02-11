# pylint: disable=no-self-use
from numpy.testing import assert_almost_equal
import torch
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.reading_comprehension.util import get_best_span


class TestRcUtil(AllenNlpTestCase):
    def test_get_best_span(self):
        span_begin_probs = torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]]).log()
        span_end_probs = torch.FloatTensor([[0.65, 0.05, 0.2, 0.05, 0.05]]).log()
        begin_end_idxs = get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 0]])

        # When we were using exclusive span ends, this was an edge case of the dynamic program.
        # We're keeping the test to make sure we get it right now, after the switch in inclusive
        # span end.  The best answer is (1, 1).
        span_begin_probs = torch.FloatTensor([[0.4, 0.5, 0.1]]).log()
        span_end_probs = torch.FloatTensor([[0.3, 0.6, 0.1]]).log()
        begin_end_idxs = get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 1]])

        # Another instance that used to be an edge case.
        span_begin_probs = torch.FloatTensor([[0.8, 0.1, 0.1]]).log()
        span_end_probs = torch.FloatTensor([[0.8, 0.1, 0.1]]).log()
        begin_end_idxs = get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 0]])

        span_begin_probs = torch.FloatTensor([[0.1, 0.2, 0.05, 0.3, 0.25]]).log()
        span_end_probs = torch.FloatTensor([[0.1, 0.2, 0.5, 0.05, 0.15]]).log()
        begin_end_idxs = get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 2]])
