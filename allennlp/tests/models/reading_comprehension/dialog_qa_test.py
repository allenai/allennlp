# pylint: disable=no-self-use,invalid-name

<<<<<<< HEAD
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch


=======
>>>>>>> 5fae856f9e827cb62fc1132622af06a6e0acd740
class DialogQATest(ModelTestCase):
    def setUp(self):
        super(DialogQATest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'dialog_qa' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'quac_sample.json')
        self.batch = Batch(self.instances)
        self.batch.index_instances(self.vocab)
        print(self.instances)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "best_span_str" in output_dict and "loss" in output_dict
        assert "followup" in output_dict and "yesno" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_batch_predictions_are_consistent(self):
<<<<<<< HEAD
        # Save some state.
=======
>>>>>>> 5fae856f9e827cb62fc1132622af06a6e0acd740
        self.ensure_batch_predictions_are_consistent()
