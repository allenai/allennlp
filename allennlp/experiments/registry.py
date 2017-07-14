from typing import Dict, List, Type

from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader


def _registry_decorator(registry_name: str, registry_dict: Dict[str, Type]):
    def decorator(name: str):
        def class_decorator_fn(cls: Type):
            if name in registry_dict:
                message = "Cannot register %s as a %s name; name already in use for %s" % (
                        name, registry_name, registry_dict[name].__name__)
                raise ConfigurationError(message)
            registry_dict[name] = cls
            return cls
        return class_decorator_fn
    return decorator


class Registry:
    _dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
    register_dataset_reader = _registry_decorator("dataset reader", _dataset_readers)

    @classmethod
    def get_dataset_readers(cls) -> List[str]:
        return list(cls._dataset_readers.keys())

    @classmethod
    def get_dataset_reader(cls, name: str) -> Type[DatasetReader]:
        return cls._dataset_readers[name]
