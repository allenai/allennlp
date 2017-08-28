from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.models import Model

class ModelTestCase(AllenNlpTestCase):
    def set_up_model(self, param_file, dataset_file):
        self.param_file = param_file
        params = Params.from_file(self.param_file)

        reader = DatasetReader.from_params(params['dataset_reader'])
        dataset = reader.read(dataset_file)
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = reader._token_indexers  # pylint: disable=protected-access

        self.model = Model.from_params(self.vocab, params['model'])
