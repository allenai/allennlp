from typing import Dict

import torch

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models import load_archive, Model

SEQUENCE_TAGGING_DATA_PATH = str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")


@Model.register("constant")
class ConstantModel(Model):
    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        return {"class": torch.tensor(98)}


class TestTrain(AllenNlpTestCase):
    def test_train_model(self):
        params = lambda: Params(
            {
                "model": {"type": "constant"},
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                "data_loader": {"batch_size": 2},
                "trainer": {"type": "no_op"},
            }
        )

        serialization_dir = self.TEST_DIR / "serialization_directory"
        train_model(params(), serialization_dir=serialization_dir)
        archive = load_archive(str(serialization_dir / "model.tar.gz"))
        model = archive.model
        assert model.forward(torch.tensor([1, 2, 3]))["class"] == torch.tensor(98)
        assert model.vocab.get_vocab_size() == 9
