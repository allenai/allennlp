import os
from typing import Any, Dict, Tuple

from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import Trainer


@Trainer.register("no_op")
class NoOpTrainer(Trainer):
    def __init__(self, serialization_dir: str, model: Model) -> None:
        """
        A trivial trainer to assist in making model archives for models that do not actually
        require training. For instance, a majority class baseline.
        """

        super().__init__(serialization_dir, cuda_device=-1)
        self.model = model

    def train(self) -> Dict[str, Any]:
        self.model.vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))

        checkpointer = Checkpointer(self._serialization_dir)
        checkpointer.save_checkpoint(epoch=0, trainer=self, is_best_so_far=True)
        return {}

    def prep_state_for_checkpointing(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.model.state_dict(), {}

    def restore_state_after_checkpointing(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass
