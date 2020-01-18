from typing import Iterable, Dict, NamedTuple, Callable, List
from collections import defaultdict
import inspect
import logging

from allennlp.training.trainer_base import TrainerBase
from allennlp.training.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class EventHandler(NamedTuple):
    name: str
    callback: Callback
    handler: Callable[[TrainerBase], None]
    priority: int


def _is_event_handler(member) -> bool:
    return inspect.ismethod(member) and hasattr(member, "_event") and hasattr(member, "_priority")


class CallbackHandler:
    """
    A `CallbackHandler` owns zero or more `Callback`s, each of which is associated
    with some "event". It then exposes a `fire_event` method, which calls each
    callback associated with that event ordered by their priorities.

    The callbacks take no parameters; instead they read from and write to this handler's
    `state`, which should be a Trainer.

    # Parameters

    callbacks : `Iterable[Callback]`
        The callbacks to be handled.
    state : `TrainerBase`
        The trainer from which the callbacks will read state
        and to which the callbacks will write state.
    verbose : bool, optional (default = False)
        If true, will log every event -> callback. Please only
        use this for debugging purposes.
    """

    def __init__(
        self, callbacks: Iterable[Callback], state: TrainerBase, verbose: bool = False
    ) -> None:
        # Set up callbacks
        self._callbacks: Dict[str, List[EventHandler]] = defaultdict(list)

        # This is just so we can find specific types of callbacks.
        self._callbacks_by_type: Dict[type, List[Callback]] = defaultdict(list)
        self.state = state
        self.verbose = verbose

        for callback in callbacks:
            self.add_callback(callback)

    def callbacks(self) -> List[Callback]:

        """
        Returns the callbacks associated with this handler.
        Each callback may be registered under multiple events,
        but we make sure to only return it once. If `typ` is specified,
        only returns callbacks of that type.
        """
        return list(
            {
                callback.callback
                for callback_list in self._callbacks.values()
                for callback in callback_list
            }
        )

    def add_callback(self, callback: Callback) -> None:
        for name, method in inspect.getmembers(callback, _is_event_handler):
            event = getattr(method, "_event")
            priority = getattr(method, "_priority")
            self._callbacks[event].append(EventHandler(name, callback, method, priority))
            self._callbacks[event].sort(key=lambda eh: eh.priority)

            self._callbacks_by_type[type(callback)].append(callback)

    def fire_event(self, event: str) -> None:
        """
        Runs every callback registered for the provided event,
        ordered by their priorities.
        """
        for event_handler in self._callbacks.get(event, []):
            if self.verbose:
                logger.info(f"event {event} -> {event_handler.name}")
            event_handler.handler(self.state)
