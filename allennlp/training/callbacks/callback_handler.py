"""
blah
"""
from typing import Iterable, Dict, NamedTuple, Callable, List
from collections import defaultdict
import inspect

from allennlp.training.callbacks.callback import Callback


class CallbackMember(NamedTuple):
    callback: Callback
    name: str
    member: Callable
    priority: int

def _is_event_handler(member) -> bool:
    return inspect.ismethod(member) and hasattr(member, '_event') and hasattr(member, '_priority')


class CallbackHandler:
    def __init__(self, callbacks: Iterable[Callback], state) -> None:
        # Set up callbacks
        self._callbacks: Dict[str, List[CallbackMember]] = defaultdict(list)
        self.state = state

        for callback in callbacks:
            self.add_callback(callback)

    def callbacks(self) -> Iterable[Callback]:
        for callback_list in self._callbacks.values():
            for callback in callback_list:
                yield callback.callback

    def add_callback(self, callback: Callback) -> None:
        for name, method in inspect.getmembers(callback, _is_event_handler):
            event = getattr(method, '_event')
            priority = getattr(method, '_priority')
            self._callbacks[event].append(CallbackMember(callback, name, method, priority))
            self._callbacks[event].sort(key=lambda cm: cm.priority)

    def fire_event(self, event: str) -> None:
        for member in self._callbacks.get(event, []):
            member.member(self.state)

    def fire_sequence(self, event: str) -> None:
        self.fire_event(f"BEFORE_{event}")
        self.fire_event(event)
        self.fire_event(f"AFTER_{event}")
