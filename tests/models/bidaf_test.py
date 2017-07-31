from allennlp.common import Params, constants
from allennlp.common.tensor import arrays_to_variables
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SquadReader
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

        self.model = BidirectionalAttentionFlow.from_params(self.vocab, Params({}))
        initializer = InitializerApplicator()
        initializer(self.model)

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_arrays())

        _ = self.model.forward(**training_arrays)
