from collections import defaultdict
from typing import List, Dict, TypeVar, Type, Generic  # pylint: disable=unused-import

from allennlp.data.dataset import Dataset
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

T = TypeVar('T')

class Registry0:
    _registry = defaultdict(dict)  # type: Dict[type, Dict[str, type]]

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registry0._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = "Cannot register %s; name already in use for %s" % (
                        name, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        return Registry0._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(Registry0._registry[cls].keys())

class DatasetReader(Registry0):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.  All
    parameters necessary to read the data apart from the filepath should be passed to the
    constructor of the ``DatasetReader``.
    """
    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
