from typing import TypeVar, Generic

from allennlp.common.registrable import Registrable

State = TypeVar('State')  # pylint: disable=invalid-name

class Callback(Registrable, Generic[State]):
    # Lower priority comes first
    priority = 0

    def __call__(self, event: str, state: State) -> None:
        raise NotImplementedError

    def get_training_state(self) -> dict:
        return {}

    def restore_training_state(self, training_state: dict) -> None:
        pass
