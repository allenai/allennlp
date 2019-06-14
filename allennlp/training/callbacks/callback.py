from typing import Callable

from allennlp.common.registrable import Registrable

def handle_event(event: str, priority: int = 0):
    def wrapper(method: Callable[[], None]):
        setattr(method, '_event', event)
        setattr(method, '_priority', priority)
        return method

    return wrapper


class Callback(Registrable):
    """
    The base class for Callbacks that are used by the CallbackTrainer.
    Notice that other than serializing / deserializing training state,
    there is no other "API".

    In a subclass you would register methods to handle specific events
    using the ``handle_event`` decorator defined above; for example

    ::

        @handle_event(Events.EPOCH_END)
        def epoch_end_stuff(self, trainer) -> None:
            ...

        @handle_event(Events.TRAINING_END)
        def training_end_stuff(self, trainer) -> None:
            ...

    In this way, each callback can respond to whatever events it wants.
    Notice also that the methods take only the trainer as input and return nothing,
    which means that any shared state needs to belong to the trainer itself.
    (Each callback can of course maintain its own non-shared state.)
    """
    def get_training_state(self) -> dict:
        """
        If this callback contains state that should be checkpointed for training,
        return it here (with a key that's unique to this callback).
        If the state lives in a pytorch object with a `state_dict`
        method, this should return the output of `state_dict()`, not the object itself.

        This default implementation suffices when there's no state to checkpoint.
        """
        # pylint: disable=no-self-use
        return {}

    def restore_training_state(self, training_state: dict) -> None:
        """
        Given a dict of training state, pull out the relevant parts
        and rehydrate the state of this callback however is necessary.

        This default implementation suffices when there's no state to restore.
        """
        pass
