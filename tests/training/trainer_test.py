# pylint: disable=invalid-name
import torch
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.trainer import Trainer
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader


class TestTrainer(AllenNlpTestCase):
    def setUp(self):
        super(TestTrainer, self).setUp()
        dataset = SequenceTaggingDatasetReader().read('tests/fixtures/data/sequence_tagging.tsv')
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

    def test_trainer_can_run(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.dataset, num_epochs=2)
        trainer.train()

    def test_trainer_can_resume_training(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.dataset,
                          num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.train()
        new_trainer = Trainer(self.model, self.optimizer,
                              self.iterator, self.dataset,
                              num_epochs=3, serialization_dir=self.TEST_DIR)

        epoch = new_trainer._restore_checkpoint()  # pylint: disable=protected-access
        assert epoch == 0
        new_trainer.train()

    def test_train_driver_raises_on_model_with_no_loss_key(self):

        class FakeModel(torch.nn.Module):
            def forward(self, **kwargs):  # pylint: disable=arguments-differ,unused-argument
                return {}
        with pytest.raises(ConfigurationError):
            trainer = Trainer(FakeModel(), self.optimizer,
                              self.iterator, self.dataset,
                              num_epochs=2, serialization_dir=self.TEST_DIR)
            trainer.train()
