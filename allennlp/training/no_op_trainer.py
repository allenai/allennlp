import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import Trainer


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
        checkpointer.save_checkpoint(epoch=0, trainer=self, is_best_so_far=True)
        return {}

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        yield self.model.state_dict(), {}
