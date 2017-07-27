# pylint: disable=invalid-name
import torch
import pytest

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.experiments.drivers.train_driver import TrainDriver
from allennlp.experiments.driver import Driver
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader


class TestTrainDriver(AllenNlpTestCase):

    def setUp(self):
        super(TestTrainDriver, self).setUp()
        self.write_sequence_tagging_data()
        dataset = SequenceTaggingDatasetReader().read(self.TRAIN_FILE)
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.model_params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "stacked_encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(self.vocab, self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01)
        self.iterator = BasicIterator(batch_size=2)

    def test_train_driver_can_run(self):
        driver = TrainDriver(self.model, self.optimizer,
                             self.iterator, self.dataset, num_epochs=2)
        driver.run()

    def test_train_driver_can_build_from_params(self):
        trainer_params = {
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                        }
                                },
                        "stacked_encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                                }
                        },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": self.TRAIN_FILE,
                "iterator": {"type": "basic"},
                "optimizer": "adam"
        }
        driver = Driver.from_params(Params(trainer_params))
        driver.run()

    def test_train_driver_can_resume_training(self):
        driver = TrainDriver(self.model, self.optimizer,
                             self.iterator, self.dataset,
                             num_epochs=1, serialization_prefix=self.TEST_DIR)
        driver.run()
        new_driver = TrainDriver(self.model, self.optimizer,
                                 self.iterator, self.dataset,
                                 num_epochs=3, serialization_prefix=self.TEST_DIR)

        epoch = new_driver._restore_checkpoint()  # pylint: disable=protected-access
        assert epoch == 0
        new_driver.run()

    def test_train_driver_raises_on_model_with_no_loss_key(self):

        class FakeModel(torch.nn.Module):
            def forward(self, **kwargs):  # pylint: disable=arguments-differ,unused-argument
                return {}
        with pytest.raises(ConfigurationError):
            driver = TrainDriver(FakeModel(), self.optimizer,
                                 self.iterator, self.dataset,
                                 num_epochs=2, serialization_prefix=self.TEST_DIR)
            driver.run()
