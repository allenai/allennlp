# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params, constants
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.models import Model, BidirectionalAttentionFlow
from allennlp.nn.util import arrays_to_variables
from allennlp.common.testing import AllenNlpTestCase


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
        dataset = SquadReader.from_params(reader_params).read('tests/fixtures/data/squad.json')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = {'tokens': SingleIdTokenIndexer(),
                               'token_characters': TokenCharactersIndexer()}

        self.model = BidirectionalAttentionFlow.from_params(self.vocab, Params({}))

        small_params = Params({
                'text_field_embedder': {
                        'tokens': {
                                'type': 'embedding',
                                'pretrained_file': constants.GLOVE_PATH,
                                'trainable': False,
                                'projection_dim': 4
                                },
                        'token_characters': {
                                'type': 'character_encoding',
                                'embedding': {
                                        'embedding_dim': 8
                                        },
                                'encoder': {
                                        'type': 'cnn',
                                        'embedding_dim': 8,
                                        'num_filters': 4,
                                        'ngram_filter_sizes': [5]
                                        }
                                }
                        },
                'phrase_layer': {
                        'type': 'lstm',
                        'bidirectional': True,
                        'input_size': 8,
                        'hidden_size': 4,
                        'num_layers': 1,
                        },
                'similarity_function': {
                        'type': 'linear',
                        'combination': 'x,y,x*y',
                        'tensor_1_dim': 8,
                        'tensor_2_dim': 8
                        },
                'modeling_layer': {
                        'type': 'lstm',
                        'bidirectional': True,
                        'input_size': 32,
                        'hidden_size': 4,
                        'num_layers': 1,
                        },
                'span_end_encoder': {
                        'type': 'lstm',
                        'bidirectional': True,
                        'input_size': 56,
                        'hidden_size': 4,
                        'num_layers': 1,
                        },

                })
        self.small_model = BidirectionalAttentionFlow.from_params(self.vocab, small_params)

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_array_dict())
        _ = self.model.forward(**training_arrays)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.small_model, self.dataset)

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

    def test_model_load(self):
        params = Params.from_file('tests/fixtures/bidaf/experiment.json')
        model = Model.load(params)

        assert isinstance(model, BidirectionalAttentionFlow)
