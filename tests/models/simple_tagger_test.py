# pylint: disable=invalid-name
from flaky import flaky
import numpy

from allennlp.common.testing import ModelTestCase
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.nn.util import arrays_to_variables


class SimpleTaggerTest(ModelTestCase):
    def setUp(self):
        super(SimpleTaggerTest, self).setUp()
        self.set_up_model('tests/fixtures/simple_tagger/experiment.json',
                          'tests/fixtures/data/sequence_tagging.tsv')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_arrays = self.dataset.as_array_dict()
        _ = self.model.forward(**arrays_to_variables(training_arrays))

    def test_tag_returns_distributions_per_token(self):
        text = TextField(["This", "is", "a", "sentence"], token_indexers={"tokens": SingleIdTokenIndexer()})
        output = self.model.tag(text)
        possible_tags = self.vocab.get_index_to_token_vocabulary("labels").values()
        for tag in output["tags"]:
            assert tag in possible_tags
        # Predictions are a distribution.
        numpy.testing.assert_almost_equal(numpy.sum(output["class_probabilities"], -1),
                                          numpy.array([1, 1, 1, 1]))
