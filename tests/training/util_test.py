import logging
import os

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.models import Model
from allennlp.training.util import make_vocab_from_params, get_metrics


logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def train_util_test_reader():
    @DatasetReader.register("train-util-test-reader")
    class TrainUtilTestReader(DatasetReader):
        def _read(self, data_path):
            logger.info("...train-util-test-reader reading from %s", data_path)
            for i in range(10):
                yield self.text_to_instance(i)

        def text_to_instance(self, index: int) -> Instance:  # type: ignore
            return Instance({"index": LabelField(index, skip_indexing=True)})

    yield TrainUtilTestReader

    del DatasetReader._registry[DatasetReader]["train-util-test-reader"]


class TestMakeVocabFromParams(AllenNlpTestCase):
    @pytest.mark.parametrize(
        "params",
        [
            Params(
                {
                    "dataset_reader": {"type": "train-util-test-reader"},
                    "train_data_path": "path-to-training-file",
                    "validation_data_path": "path-to-validation-file",
                    "test_data_path": "path-to-validation-file",
                    "datasets_for_vocab_creation": [],
                }
            ),
            Params(
                {
                    "dataset_reader": {"type": "train-util-test-reader"},
                    "train_data_path": "path-to-training-file",
                    "datasets_for_vocab_creation": [],
                }
            ),
            Params(
                {
                    "dataset_reader": {"type": "train-util-test-reader"},
                    "train_data_path": "path-to-training-file",
                    "validation_data_path": "path-to-validation-file",
                    "test_data_path": "path-to-validation-file",
                    "vocabulary": {"type": "empty"},
                }
            ),
        ],
    )
    def test_no_instances_read_for_vocab(self, caplog, params):
        _ = make_vocab_from_params(params, str(self.TEST_DIR))
        log_messages = "\n".join([rec.message for rec in caplog.records])
        assert "...train-util-test-reader reading from" not in log_messages
        assert "Reading training data" not in log_messages
        assert "Reading validation data" not in log_messages
        assert "Reading test data" not in log_messages

    def test_only_train_read_for_vocab(self, caplog):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
            }
        )
        _ = make_vocab_from_params(params, str(self.TEST_DIR))
        log_messages = "\n".join([rec.message for rec in caplog.records])
        assert "...train-util-test-reader reading from path-to-training-file" in log_messages
        assert "...train-util-test-reader reading from path-to-validation-file" not in log_messages
        assert "...train-util-test-reader reading from path-to-test-file" not in log_messages
        assert "Reading training data" in log_messages
        assert "Reading validation data" not in log_messages
        assert "Reading test data" not in log_messages

    def test_all_datasets_read_for_vocab(self, caplog):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
                "validation_data_path": "path-to-validation-file",
                "test_data_path": "path-to-test-file",
            }
        )
        _ = make_vocab_from_params(params, str(self.TEST_DIR))
        log_messages = "\n".join([rec.message for rec in caplog.records])
        assert "...train-util-test-reader reading from path-to-training-file" in log_messages
        assert "...train-util-test-reader reading from path-to-validation-file" in log_messages
        assert "...train-util-test-reader reading from path-to-test-file" in log_messages
        assert "Reading training data" in log_messages
        assert "Reading validation data" in log_messages
        assert "Reading test data" in log_messages

    def test_only_specified_datasets_read_for_vocab(self, caplog):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
                "validation_data_path": "path-to-validation-file",
                "test_data_path": "path-to-test-file",
                "datasets_for_vocab_creation": ["train", "validation"],
            }
        )
        _ = make_vocab_from_params(params, str(self.TEST_DIR))
        log_messages = "\n".join([rec.message for rec in caplog.records])
        assert "...train-util-test-reader reading from path-to-training-file" in log_messages
        assert "...train-util-test-reader reading from path-to-validation-file" in log_messages
        assert "...train-util-test-reader reading from path-to-test-file" not in log_messages
        assert "Reading training data" in log_messages
        assert "Reading validation data" in log_messages
        assert "Reading test data" not in log_messages

    def test_using_seperate_validation_reader(self, caplog):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "validation_dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
                "validation_data_path": "path-to-validation-file",
            }
        )
        _ = make_vocab_from_params(params, str(self.TEST_DIR))
        log_messages = "\n".join([rec.message for rec in caplog.records])
        assert "Using a separate dataset reader to load validation and test data" in log_messages

    def test_invalid_datasets_for_vocab_creation(self):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
                "validation_data_path": "path-to-validation-file",
                "datasets_for_vocab_creation": ["train", "validation", "test"],
            }
        )
        with pytest.raises(ConfigurationError, match="invalid 'datasets_for_vocab_creation' test"):
            make_vocab_from_params(params, str(self.TEST_DIR))

    def test_raise_error_if_directory_non_empty(self):
        params = Params(
            {
                "dataset_reader": {"type": "train-util-test-reader"},
                "train_data_path": "path-to-training-file",
                "validation_data_path": "path-to-validation-file",
            }
        )
        os.makedirs(self.TEST_DIR / "vocabulary")
        with open(self.TEST_DIR / "vocabulary" / "blah", "w") as random_file:
            random_file.write("BLAH!")
        with pytest.raises(ConfigurationError, match="The 'vocabulary' directory in the provided"):
            make_vocab_from_params(params, str(self.TEST_DIR))

    def test_get_metrics(self):
        class FakeModel(Model):
            def forward(self, **kwargs):
                return {}

        model = FakeModel(None)
        total_loss = 100.0
        batch_loss = 10.0
        num_batches = 2
        metrics = get_metrics(model, total_loss, None, batch_loss, None, num_batches)

        assert metrics["loss"] == float(total_loss / num_batches)
        assert metrics["batch_loss"] == batch_loss

        metrics = get_metrics(model, total_loss, None, None, None, num_batches)

        assert metrics["loss"] == float(total_loss / num_batches)
        assert "batch_loss" not in metrics
