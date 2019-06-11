from allennlp.common.registrable import Registrable

def handle_event(event: str, priority: int = 0):
    def wrapper(method):
        setattr(method, '_event', event)
        setattr(method, '_priority', priority)
        return method

    return wrapper

class Callback(Registrable):
    # Lower priority comes first
    priority = 0

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
