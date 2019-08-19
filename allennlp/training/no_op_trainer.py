import os
from typing import Dict, Any

from allennlp.common import Params
from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer_pieces import TrainerPieces


@TrainerBase.register("no_op")
class NoOpTrainer(TrainerBase):
    def __init__(self, serialization_dir: str, model: Model) -> None:
        """
        A trivial trainer to assist in making model archives for models that do not actually
        require training. For instance, a majority class baseline.
        """

        super().__init__(serialization_dir, cuda_device=-1)
        self.model = model

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        # pylint: disable=arguments-differ
        pieces = TrainerPieces.from_params(params, serialization_dir, recover, cache_directory, cache_prefix)  # pylint: disable=no-member
        return NoOpTrainer(serialization_dir, pieces.model)

    def train(self) -> Dict[str, Any]:
        self.model.vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))

        checkpointer = Checkpointer(self._serialization_dir)
        checkpointer.save_checkpoint(epoch=0,
                                     model_state=self.model.state_dict(),
                                     training_states={},
                                     is_best_so_far=True)
        return {}
