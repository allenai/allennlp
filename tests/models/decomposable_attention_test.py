# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import numpy
from numpy.testing import assert_almost_equal

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.fields import TextField
from allennlp.models import DecomposableAttention, Model
from allennlp.nn.util import arrays_to_variables


class TestDecomposableAttention(ModelTestCase):
    def setUp(self):
        super(TestDecomposableAttention, self).setUp()
        self.set_up_model('tests/fixtures/decomposable_attention/experiment.json',
                          'tests/fixtures/data/snli.jsonl')

    def test_forward_pass_runs_correctly(self):
        training_arrays = arrays_to_variables(self.dataset.as_array_dict())
        _ = self.model.forward(**training_arrays)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_predict_entailment_gives_reasonable_outputs(self):
        premise = TextField(["A", "dog", "is", "a", "mammal"], token_indexers=self.token_indexers)
        hypothesis = TextField(["A", "dog", "is", "an", "animal"], token_indexers=self.token_indexers)
        output_dict = self.model.predict_entailment(premise, hypothesis)
        assert_almost_equal(numpy.sum(output_dict["label_probs"], -1), 1, decimal=6)

    def test_model_load(self):
        params = Params.from_file('tests/fixtures/decomposable_attention/experiment.json')
        model = Model.load(params, serialization_dir='tests/fixtures/decomposable_attention/serialization')

        assert isinstance(model, DecomposableAttention)
