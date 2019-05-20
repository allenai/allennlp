from typing import List, Generic, TypeVar
from typing_extensions import Protocol

from allennlp.common.registrable import Registrable

# TODO(joelgrus): the version of mypy we're pinned on doesn't do
# anything with typing_extensions
State = TypeVar('State', bound=Protocol)  # pylint: disable=invalid-name

class Callback(Registrable, Generic[State]):
    # Lower priority comes first
    priority = 0
    state_requires: List[str] = []

    def __call__(self, event: str, state: State) -> None:
        raise NotImplementedError
