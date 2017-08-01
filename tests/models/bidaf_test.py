import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params, constants
from allennlp.common.tensor import arrays_to_variables
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.models import BidirectionalAttentionFlow
from allennlp.training.initializers import InitializerApplicator
from allennlp.testing.test_case import AllenNlpTestCase


class BidirectionalAttentionFlowTest(AllenNlpTestCase):
    def setUp(self):
        super(BidirectionalAttentionFlowTest, self).setUp()

        constants.GLOVE_PATH = 'tests/fixtures/glove.6B.100d.sample.txt.gz'
        reader_params = Params({
                'token_indexers': {
                        'tokens': {
                                'type': 'single_id'
                                },
                        'token_characters': {
                                'type': 'characters'
                                }
                        }
                })
        dataset = SquadReader.from_params(reader_params).read('tests/fixtures/squad_example.json')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = {'tokens': SingleIdTokenIndexer(),
                               'token_characters': TokenCharactersIndexer()}

        self.model = BidirectionalAttentionFlow.from_params(self.vocab, Params({}))
        initializer = InitializerApplicator()
        initializer(self.model)

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_arrays())

        _ = self.model.forward(**training_arrays)

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


    def test_get_best_span(self):
        # Note that the best span cannot be (1, 0) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = torch.FloatTensor([0.1, 0.3, 0.05, 0.3, 0.25])
        span_end_probs = torch.FloatTensor([0.5, 0.1, 0.2, 0.05, 0.15])
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs,
                                                                   span_end_probs)
        assert begin_end_idxs == (1, 2)

        # Testing an edge case of the dynamic program here, for the order of when you update the
        # best previous span position.  We should not get (1, 1), because that's an empty span.
        span_begin_probs = torch.FloatTensor([0.4, 0.5, 0.1])
        span_end_probs = torch.FloatTensor([0.3, 0.6, 0.1])
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs,
                                                                   span_end_probs)
        assert begin_end_idxs == (0, 1)

        # test higher-order input
        # Note that the best span cannot be (1, 1) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]])
        span_end_probs = torch.FloatTensor([[0.1, 0.5, 0.2, 0.05, 0.15]])
        begin_end_idxs = BidirectionalAttentionFlow._get_best_span(span_begin_probs,
                                                                   span_end_probs)
        assert begin_end_idxs == (1, 2)
