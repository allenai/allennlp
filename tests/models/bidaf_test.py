# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.fields import TextField
from allennlp.models import BidirectionalAttentionFlow
from allennlp.nn.util import arrays_to_variables


class BidirectionalAttentionFlowTest(ModelTestCase):
    def setUp(self):
        super(BidirectionalAttentionFlowTest, self).setUp()
        self.set_up_model('tests/fixtures/bidaf/experiment.json', 'tests/fixtures/data/squad.json')

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_array_dict())
        _ = self.model.forward(**training_arrays)
        metrics = self.model.get_metrics(reset=True)
        # We've set up the data such that there's a fake answer that consists of the whole
        # paragraph.  _Any_ valid prediction for that question should produce an F1 of greater than
        # zero, while if we somehow haven't been able to load the evaluation data, or there was an
        # error with using the evaluation script, this will fail.  This makes sure that we've
        # loaded the evaluation data correctly and have hooked things up to the official evaluation
        # script.
        assert metrics['f1'] > 0

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_predict_span_gives_reasonable_outputs(self):
        # TODO(mattg): "What", "is", "?" crashed, because the CNN encoder expected at least 5
        # characters.  We need to fix that somehow.
        question = TextField(["Whatever", "is", "?"], token_indexers=self.token_indexers)
        passage = TextField(["This", "is", "a", "passage", SquadReader.STOP_TOKEN],
                            token_indexers=self.token_indexers)
        output_dict = self.model.predict_span(question, passage)

        assert_almost_equal(numpy.sum(output_dict["span_start_probs"], -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(output_dict["span_end_probs"], -1), 1, decimal=6)

        span_start, span_end = output_dict['best_span']
        assert span_start >= 0
        assert span_start < span_end
        assert span_end < passage.sequence_length()

        assert isinstance(output_dict['best_span_str'], str)

    def test_get_best_span(self):
        # pylint: disable=protected-access

        # Note that the best span cannot be (1, 0) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = Variable(torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.5, 0.1, 0.2, 0.05, 0.15]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 2]])

        # Testing an edge case of the dynamic program here, for the order of when you update the
        # best previous span position.  We should not get (1, 1), because that's an empty span.
        span_begin_probs = Variable(torch.FloatTensor([[0.4, 0.5, 0.1]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.3, 0.6, 0.1]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 1]])

        # Testing another edge case of the dynamic program here, where (0, 0) is the best solution
        # without constraints.
        span_begin_probs = Variable(torch.FloatTensor([[0.8, 0.1, 0.1]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.8, 0.1, 0.1]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 1]])

        # test higher-order input
        # Note that the best span cannot be (1, 1) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = Variable(torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.1, 0.5, 0.2, 0.05, 0.15]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 2]])
