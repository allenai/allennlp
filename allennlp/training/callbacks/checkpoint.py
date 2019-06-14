from typing import List
import traceback

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events


@Callback.register("checkpoint")
class CheckpointCallback(Callback):
    """
    Callback that orchestrates checkpointing of your model and training state.

    Parameters
    ----------
    checkpointer : ``Checkpointer``
        The checkpoint reader and writer to use.
    state_dict_attrs : ``List[str]``, optional (default = ['optimizer'])
        The attributes of the Trainer state whose `.state_dict()`
        should be persisted at each checkpoint.
    other_attrs : ``List[str]``, optional (default = ['batch_num_total'])
        The attributes of the Trainer state that should be persisted
        as-is at each checkpoint.
    """
    def __init__(self,
                 checkpointer: Checkpointer,
                 state_dict_attrs: List[str] = None,
                 other_attrs: List[str] = None) -> None:
        self.checkpointer = checkpointer
        self.state_dict_attrs = state_dict_attrs or ['optimizer']
        self.other_attrs = other_attrs or ['batch_num_total']

    @handle_event(Events.SAVE_CHECKPOINT)
    def save_checkpoint(self, trainer):
        training_states = {}

        # Add state_dict attributes
        for attr in self.state_dict_attrs:
            state_attr = getattr(trainer, attr)
            if state_attr is not None:
                training_states[attr] = state_attr.state_dict()

        # Add other attributes
        for attr in self.other_attrs:
            training_states[attr] = getattr(trainer, attr)

        # Get attributes from callbacks
        for callback in trainer.handler.callbacks():
            training_states.update(callback.get_training_state())

        is_best_so_far = training_states.pop("is_best_so_far", True)
        self.checkpointer.save_checkpoint(
                model_state=trainer.model.state_dict(),
                epoch=trainer.checkpoint_epoch,
                training_states=training_states,
                is_best_so_far=is_best_so_far)

    @handle_event(Events.RESTORE_CHECKPOINT)
    def restore_checkpoint(self, trainer):
        # Restores the model and training state from the last saved checkpoint.
        # This includes an epoch count and optimizer state, which is serialized separately
        # from model parameters. This function should only be used to continue training -
        # if you wish to load a model for inference/load parts of a model into a new
        # computation graph, you should use the native Pytorch functions:
        # `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        # If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        # this will do nothing.
        try:
            model_state, training_state = self.checkpointer.restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  "
                                     "Did you mean to output to a different serialization directory "
                                     "or delete the existing serialization directory?")

        if not training_state:
            # No checkpoint to restore, start at 0
            trainer.epoch_number = 0
            return

        trainer.model.load_state_dict(model_state)

        # Restore state_dict attrs
        for attr in self.state_dict_attrs:
            state_attr = getattr(trainer, attr)
            if state_attr is not None:
                state_attr.load_state_dict(training_state[attr])

        # Restore other attrs
        for attr in self.other_attrs:
            setattr(trainer, attr, training_state[attr])

        # Restore callback attrs
        for callback in trainer.handler.callbacks():
            callback.restore_training_state(training_state)

        if isinstance(training_state["epoch"], int):
            trainer.epoch_number = training_state["epoch"] + 1
        else:
            trainer.epoch_number = int(training_state["epoch"].split('.')[0]) + 1

    @handle_event(Events.TRAINING_END)
    def load_best_model_state(self, trainer):
        # Load the best model state before returning
        best_model_state = self.checkpointer.best_model_state()
        if best_model_state:
            trainer.model.load_state_dict(best_model_state)

    @classmethod
    def from_params(cls, params: Params, serialization_dir: str) -> 'CheckpointCallback':  # type: ignore
        # pylint: disable=arguments-differ
        checkpointer_params = params.pop("checkpointer", None)
        if checkpointer_params:
            checkpointer = Checkpointer.from_params(checkpointer_params, serialization_dir=serialization_dir)
        else:
            checkpointer = Checkpointer(serialization_dir=serialization_dir)

        state_dict_attrs = params.pop("state_dict_attrs", None)
        other_attrs = params.pop("other_attrs", None)

        return CheckpointCallback(checkpointer, state_dict_attrs, other_attrs)
