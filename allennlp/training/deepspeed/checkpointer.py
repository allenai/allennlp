from typing import Union, Dict, Any, List, Tuple, Optional

import logging
import os
import re
import shutil
import time

from pathlib import Path

import torch

import allennlp
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util, Checkpointer

logger = logging.getLogger(__name__)
_DeepspeedTrainer = "allennlp.training.deepspeed.trainer.DeepspeedTrainer"


class DeepspeedCheckpointer(Checkpointer):
    def save_checkpoint(
        self,
        epoch: Union[int, str],
        trainer: _DeepspeedTrainer,
        is_best_so_far: bool = False,
        save_model_only=False,
    ) -> None:
        if self._serialization_dir is None:
            return

        with trainer.get_checkpoint_state() as state:
            model_engine, model_state, training_states = state
            
            checkpoint_id = "deepspeed_epoch_{}".format(epoch)
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}".format(epoch))
            model_engine.save_checkpoint(self._serialization_dir, checkpoint_id)

            # TODO
            # Model will need a weight file to load; 
            # not sure if ZeRO stage 2 will mess this up
            if not os.path.isfile(model_path):
                    torch.save(model_state, model_path)
            if save_model_only:
                return

            training_path = os.path.join(
                self._serialization_dir, "training_state_epoch_{}.th".format(epoch)
            )
            if not os.path.isfile(training_path):
                torch.save({**training_states, "epoch": epoch}, training_path)

        # The main checkpointing logic is now done, this is just shuffling files around, to keep
        # track of best weights, and to remove old checkpoints, if desired.
        if is_best_so_far:
            logger.info(
                "Best validation performance so far. Copying weights to '%s/best.th'.",
                self._serialization_dir,
            )
            shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))
            
            engine_dir = os.path.join(self._serialization_dir, "best_deepspeed")
            shutil.rmtree(engine_dir, ignore_errors=True) # in case no previous checkpoints
            shutil.copytree(os.path.join(self._serialization_dir, checkpoint_id), engine_dir)

        if (
            self._num_serialized_models_to_keep is not None
            and self._num_serialized_models_to_keep >= 0
        ):
            self._serialized_paths.append((time.time(), model_path, training_path))
            if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                paths_to_remove = self._serialized_paths.pop(0)
                # Check to see if we should keep this checkpoint, if it has been longer
                # then self._keep_serialized_model_every_num_seconds since the last
                # kept checkpoint.
                remove_path = True
                if self._keep_serialized_model_every_num_seconds is not None:
                    save_time = paths_to_remove[0]
                    time_since_checkpoint_kept = (
                        save_time - self._last_permanent_saved_checkpoint_time
                    )
                    if (
                        time_since_checkpoint_kept
                        > self._keep_serialized_model_every_num_seconds
                    ):
                        # We want to keep this checkpoint.
                        remove_path = False
                        self._last_permanent_saved_checkpoint_time = save_time
                if remove_path:
                    for fname in paths_to_remove[1:]:
                        if os.path.isfile(fname):
                            os.remove(fname)

    def find_latest_checkpoint(self) -> Optional[Tuple[str, str]]:
        latest = super().find_latest_checkpoint()
        if not latest:
            return None

        model_path, training_state_path = latest

        checkpoints = (self._serialization_dir and Path(self._serialization_dir).glob('deepspeed_epoch_*')) or []
        checkpoints = sorted(c for c in checkpoints if c.is_dir())
        if not checkpoints:
            return None

        engine_path = checkpoints[-1]
        return engine_path, model_path, training_state_path

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return {}, {}, {}

        checkpoint_id, model_path, training_state_path = latest_checkpoint

        model_state = torch.load(model_path, map_location=nn_util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=nn_util.device_mapping(-1))
        return checkpoint_id, model_state, training_state