from typing import Union, Dict, Any, List, Tuple

import logging
import os
import re
import shutil
import time

import torch

from allennlp.common import Registrable
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)


class Checkpointer(Registrable):
    """
    This class implements the functionality for checkpointing your model and trainer state
    during training. It is agnostic as to what those states look like (they are typed as
    Dict[str, Any]), but they will be fed to `torch.save` so they should be serializable
    in that sense. They will also be restored as Dict[str, Any], which means the calling
    code is responsible for knowing what to do with them.
    """

    def __init__(
        self,
        serialization_dir: str = None,
        keep_serialized_model_every_num_seconds: int = None,
        num_serialized_models_to_keep: int = 20,
    ) -> None:
        self._serialization_dir = serialization_dir
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._num_serialized_models_to_keep = num_serialized_models_to_keep

        self._last_permanent_saved_checkpoint_time = time.time()
        self._serialized_paths: List[Tuple[float, str, str]] = []

    def save_checkpoint(
        self,
        epoch: Union[int, str],
        model_state: Dict[str, Any],
        training_states: Dict[str, Any],
        is_best_so_far: bool,
    ) -> None:
        if self._serialization_dir is not None:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch)
            )
            torch.save(model_state, model_path)
            training_path = os.path.join(
                self._serialization_dir, "training_state_epoch_{}.th".format(epoch)
            )
            torch.save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far:
                logger.info(
                    "Best validation performance so far. Copying weights to '%s/best.th'.",
                    self._serialization_dir,
                )
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

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

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = self._serialization_dir is not None and any(
            "model_state_epoch_" in x for x in os.listdir(self._serialization_dir)
        )

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split(".")
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), "0"])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == "0":
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = "{0}.{1}".format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(
            self._serialization_dir, "model_state_epoch_{}.th".format(epoch_to_load)
        )
        training_state_path = os.path.join(
            self._serialization_dir, "training_state_epoch_{}.th".format(epoch_to_load)
        )

        return (model_path, training_state_path)

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes a training state (typically consisting of an epoch count and optimizer state),
        which is serialized separately from  model parameters. This function should only be used to
        continue training - if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return empty dicts.

        # Returns

        states: Tuple[Dict[str, Any], Dict[str, Any]]
            The model state and the training state.
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return {}, {}

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, map_location=nn_util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=nn_util.device_mapping(-1))
        return model_state, training_state

    def best_model_state(self) -> Dict[str, Any]:
        if self._serialization_dir:
            logger.info("loading best weights")
            best_model_state_path = os.path.join(self._serialization_dir, "best.th")
            return torch.load(best_model_state_path, map_location=nn_util.device_mapping(-1))
        else:
            logger.info(
                "cannot load best weights without `serialization_dir`, "
                "so you're just getting the last weights"
            )
            return {}
