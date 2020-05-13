# pylint: disable=invalid-name,protected-access
import pathlib, json, os

from allennlp_models.nli import snli_reader
from allennlp.common.testing import ModelTestCase
from allennlp.common.testing.test_case import TEST_DIR
from allennlp.commands.train import train_model, train_model_from_file

os.environ['ARCHIVE_PATH'] = "/tmp/taskA"

class TransferLearningTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('allennlp/tests/fixtures/esnli.jsonnet',
                          'allennlp/tests/fixtures/esnli_train.jsonl')

    def test_taskA_end_to_end(self):
        train_model_from_file("allennlp/tests/transfer_learning/fixtures/esnli.jsonnet", serialization_dir="/tmp/taskA", force=True)
        
    def test_taskB_end_to_end(self):
        train_model_from_file("allennlp/tests/transfer_learning/fixtures/movies.jsonnet", serialization_dir="/tmp/taskB", force=True)
        
