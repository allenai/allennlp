from typing import Union, Dict, Any, Tuple, Optional, Iterable

import logging
import os
import shutil
from overrides import overrides

from pathlib import Path

import allennlp
from allennlp.training import Checkpointer

logger = logging.getLogger(__name__)


class DeepspeedCheckpointer(Checkpointer):
    @overrides
    def save_checkpoint(
        self,
        epoch: Union[int, str],
        trainer: "allennlp.training.deepspeed.trainer.DeepspeedTrainer",
        is_best_so_far: bool = False,
        save_model_only: bool = False,
    ) -> None:
        if self._serialization_dir is None:
            return

        super().save_checkpoint(epoch, trainer, is_best_so_far, save_model_only)

        checkpoint_id = "deepspeed_epoch_{}".format(epoch)
        trainer.model_engine.save_checkpoint(self._serialization_dir, checkpoint_id)
        if trainer._primary and is_best_so_far:
            engine_dir = os.path.join(self._serialization_dir, "best_deepspeed")
            shutil.rmtree(engine_dir, ignore_errors=True)  # in case no previous checkpoints
            shutil.copytree(os.path.join(self._serialization_dir, checkpoint_id), engine_dir)

    def find_latest_deepspeed_checkpoint(self) -> Optional[str]:
        checkpoints: Iterable[Path] = (
            self._serialization_dir and Path(self._serialization_dir).glob("deepspeed_epoch_*")
        ) or []
        checkpoints = sorted(c for c in checkpoints if c.is_dir())
        if not checkpoints:
            return None

        engine_path = str(checkpoints[-1])
        return engine_path

    @overrides
    def restore_checkpoint(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        model_state, training_state = super().restore_checkpoint()
        checkpoint_id = self.find_latest_deepspeed_checkpoint()
        return checkpoint_id, model_state, training_state
