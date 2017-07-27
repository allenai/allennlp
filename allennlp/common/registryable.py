from collections import defaultdict
from typing import TypeVar, Type, Dict, List  # pylint: disable=unused-import

from allennlp.common.checks import ConfigurationError

T = TypeVar('T')

class Registryable:
    """
    any class that inherits from this one automatically gets a 'registry' with the supplied methods.
    there is one global registry dict (of dicts) but it's keyed by the classes themselves, so there
    should be no way to conflict
    """

    _registry = defaultdict(dict)  # type: Dict[type, Dict[str, type]]

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registryable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                        name, cls.__name__, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        return Registryable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(Registryable._registry[cls].keys())
