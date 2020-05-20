import os
from typing import Dict

import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.models.model import Model
from allennlp.training import NoOpTrainer


class ConstantModel(Model):
    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        return {"class": torch.tensor(98)}


class TestNoOpTrainer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.instances = SequenceTaggingDatasetReader().read(
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
        )
        vocab = Vocabulary.from_instances(self.instances)
        self.vocab = vocab
        self.model = ConstantModel(vocab)

    def test_trainer_serializes(self):
        serialization_dir = self.TEST_DIR / "serialization_dir"
        trainer = NoOpTrainer(serialization_dir=serialization_dir, model=self.model)
        metrics = trainer.train()
        assert metrics == {}
        assert os.path.exists(serialization_dir / "best.th")
        assert os.path.exists(serialization_dir / "vocabulary")
