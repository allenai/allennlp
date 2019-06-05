from typing import TypeVar, Generic

from allennlp.common.registrable import Registrable

TrainerType = TypeVar('TrainerType')  # pylint: disable=invalid-name

class Callback(Registrable, Generic[TrainerType]):
    # Lower priority comes first
    priority = 0

    def __call__(self, event: str, trainer: TrainerType) -> None:
        raise NotImplementedError

    def get_training_state(self) -> dict:
        """
        If this callback contains state that should be checkpointed for training,
        return it here (with a key that's unique to this callback).
        If the state lives in a pytorch object with a `state_dict`
        method, this should return the output of `state_dict()`, not the object itself.
        """
        # pylint: disable=no-self-use
        return {}

    def restore_training_state(self, training_state: dict) -> None:
        """
        Given a dict of training state, pull out the relevant parts
        and rehydrate the state of this callback however is necessary.
        """
        pass
