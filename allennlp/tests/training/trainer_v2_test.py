import torch
from torch.utils.data import DataLoader
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training.trainer_v2 import TrainerV2


class TestTrainer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.instances = SequenceTaggingDatasetReader().read(
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
        )
        vocab = Vocabulary.from_instances(self.instances)
        self.vocab = vocab
        self.model_params = Params(
            {
                "text_field_embedder": {
                    "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                },
                "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
            }
        )
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.data_loader = DataLoader(self.instances, batch_size=2)
        self.validation_data_loader = DataLoader(self.instances, batch_size=2)
        self.instances.index_with(vocab)

    def test_trainer_can_run(self):
        trainer = TrainerV2(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            train_dataset=self.instances,
            validation_dataset=self.instances,
            num_epochs=2,
        )
        metrics = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)

        # Making sure that both increasing and decreasing validation metrics work.
        trainer = TrainerV2(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            train_dataset=self.instances,
            validation_dataset=self.instances,
            validation_metric="+loss",
            num_epochs=2,
        )
        metrics = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)
        assert "peak_cpu_memory_MB" in metrics
        assert isinstance(metrics["peak_cpu_memory_MB"], float)
        assert metrics["peak_cpu_memory_MB"] > 0
