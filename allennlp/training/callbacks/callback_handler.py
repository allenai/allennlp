"""
blah
"""
from typing import Iterable, TypeVar, Generic

from allennlp.common.registrable import Registrable
from allennlp.training.callbacks.callback import Callback

State = TypeVar('State')  # pylint: disable=invalid-name

class CallbackHandler(Registrable, Generic[State]):
    def __init__(self, callbacks: Iterable[Callback[State]], state: State) -> None:
        # Set up callbacks
        self.callbacks = list(callbacks or [])
        self.callbacks.sort(key=lambda cb: cb.priority)
        self.state = state

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda cb: cb.priority)

    def remove_callback(self, callback: Callback) -> None:
        self.callbacks = [cb for cb in self.callbacks if cb != callback]

    def fire_event(self, event: str) -> None:
        print(event)
        for callback in self.callbacks:
            callback(event, self.state)

    def fire_sequence(self, event: str) -> None:
        self.fire_event(f"BEFORE_{event}")
        self.fire_event(event)
        self.fire_event(f"AFTER_{event}")
