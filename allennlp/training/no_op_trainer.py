import os
from typing import Any, Dict

from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import Trainer, TrainerCheckpoint


@Trainer.register("no_op")
class NoOpTrainer(Trainer):
    """
    Registered as a `Trainer` with name "no_op".
    """

    def __init__(self, serialization_dir: str, model: Model) -> None:
        """
        A trivial trainer to assist in making model archives for models that do not actually
        require training. For instance, a majority class baseline.

        In a typical AllenNLP configuration file, neither the `serialization_dir` nor the `model`
        arguments would need an entry.
        """

        super().__init__(serialization_dir, cuda_device=-1)
        self.model = model

    def train(self) -> Dict[str, Any]:
        assert self._serialization_dir is not None
        self.model.vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))
        checkpointer = Checkpointer(self._serialization_dir)
        checkpointer.save_checkpoint(self)
        return {}

    def get_checkpoint_state(self) -> TrainerCheckpoint:
        return TrainerCheckpoint(self.model.state_dict(), {})
