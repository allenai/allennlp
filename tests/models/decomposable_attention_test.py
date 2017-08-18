# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal

from allennlp.common import Params, constants
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import DecomposableAttention, Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import arrays_to_variables
from allennlp.common.testing import AllenNlpTestCase


class TestDecomposableAttention(AllenNlpTestCase):
    def setUp(self):
        super(TestDecomposableAttention, self).setUp()

        constants.GLOVE_PATH = 'tests/fixtures/glove.6B.300d.sample.txt.gz'
        dataset = SnliReader().read('tests/fixtures/data/snli.jsonl')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

        self.model = DecomposableAttention.from_params(self.vocab, Params({}))
        initializer = InitializerApplicator()
        initializer(self.model)

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_array_dict())
        _ = self.model.forward(**training_arrays)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.model, self.dataset)

    def test_predict_entailment_gives_reasonable_outputs(self):
        premise = TextField(["A", "dog", "is", "a", "mammal"], token_indexers=self.token_indexers)
        hypothesis = TextField(["A", "dog", "is", "an", "animal"], token_indexers=self.token_indexers)
        output_dict = self.model.predict_entailment(premise, hypothesis)
        assert_almost_equal(numpy.sum(output_dict["label_probs"], -1), 1, decimal=6)

    def test_model_load(self):
        params = Params.from_file('tests/fixtures/decomposable_attention/experiment.json')
        model = Model.load(params)

        assert isinstance(model, DecomposableAttention)
