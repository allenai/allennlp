# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pytest
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import BidirectionalAttentionFlow, Model


class BidirectionalAttentionFlowTest(ModelTestCase):
    def setUp(self):
        super(BidirectionalAttentionFlowTest, self).setUp()
        self.set_up_model('tests/fixtures/bidaf/experiment.json', 'tests/fixtures/data/squad.json')

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)

        metrics = self.model.get_metrics(reset=True)
        # We've set up the data such that there's a fake answer that consists of the whole
        # paragraph.  _Any_ valid prediction for that question should produce an F1 of greater than
        # zero, while if we somehow haven't been able to load the evaluation data, or there was an
        # error with using the evaluation script, this will fail.  This makes sure that we've
        # loaded the evaluation data correctly and have hooked things up to the official evaluation
        # script.
        assert metrics['f1'] > 0

        span_start_probs = output_dict['span_start_probs'][0].data.numpy()
        span_end_probs = output_dict['span_start_probs'][0].data.numpy()
        assert_almost_equal(numpy.sum(span_start_probs, -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(span_end_probs, -1), 1, decimal=6)
        span_start, span_end = tuple(output_dict['best_span'][0].data.numpy())
        assert span_start >= 0
        assert span_start <= span_end
        assert span_end < self.instances[0].fields['passage'].sequence_length()
        assert isinstance(output_dict['best_span_str'][0], str)

    # Some recent efficiency changes (using bmm for `weighted_sum`, the more efficient
    # `masked_softmax`...) have made this _very_ flaky...
    @flaky(max_runs=5)
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    @flaky
    def test_batch_predictions_are_consistent(self):
        # The CNN encoder has problems with this kind of test - it's not properly masked yet, so
        # changing the amount of padding in the batch will result in small differences in the
        # output of the encoder.  Because BiDAF is so deep, these differences get magnified through
        # the network and make this test impossible.  So, we'll remove the CNN encoder entirely
        # from the model for this test.  If/when we fix the CNN encoder to work correctly with
        # masking, we can change this back to how the other models run this test, with just a
        # single line.
        # pylint: disable=protected-access,attribute-defined-outside-init

        # Save some state.
        saved_model = self.model
        saved_instances = self.instances

        # Modify the state, run the test with modified state.
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])
        reader._token_indexers = {'tokens': reader._token_indexers['tokens']}
        self.instances = reader.read('tests/fixtures/data/squad.json')
        vocab = Vocabulary.from_instances(self.instances)
        for instance in self.instances:
            instance.index_fields(vocab)
        del params['model']['text_field_embedder']['token_characters']
        params['model']['phrase_layer']['input_size'] = 2
        self.model = Model.from_params(vocab, params['model'])

        self.ensure_batch_predictions_are_consistent()

        # Restore the state.
        self.model = saved_model
        self.instances = saved_instances

    def test_get_best_span(self):
        # pylint: disable=protected-access

        span_begin_probs = Variable(torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.65, 0.05, 0.2, 0.05, 0.05]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 0]])

        # When we were using exlcusive span ends, this was an edge case of the dynamic program.
        # We're keeping the test to make sure we get it right now, after the switch in inclusive
        # span end.  The best answer is (1, 1).
        span_begin_probs = Variable(torch.FloatTensor([[0.4, 0.5, 0.1]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.3, 0.6, 0.1]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 1]])

        # Another instance that used to be an edge case.
        span_begin_probs = Variable(torch.FloatTensor([[0.8, 0.1, 0.1]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.8, 0.1, 0.1]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 0]])

        span_begin_probs = Variable(torch.FloatTensor([[0.1, 0.2, 0.05, 0.3, 0.25]])).log()
        span_end_probs = Variable(torch.FloatTensor([[0.1, 0.2, 0.5, 0.05, 0.15]])).log()
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs, span_end_probs)
        assert_almost_equal(begin_end_idxs.data.numpy(), [[1, 2]])

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the phrase layer wrong - it should be 10 to match
        # the embedding + char cnn dimensions.
        params["model"]["phrase_layer"]["input_size"] = 12
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

        params = Params.from_file(self.param_file)
        # Make the modeling layer input_dimension wrong - it should be 40 to match
        # 4 * output_dim of the phrase_layer.
        params["model"]["phrase_layer"]["input_size"] = 30
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

        params = Params.from_file(self.param_file)
        # Make the modeling layer input_dimension wrong - it should be 70 to match
        # 4 * phrase_layer.output_dim + 3 * modeling_layer.output_dim.
        params["model"]["span_end_encoder"]["input_size"] = 50
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))
