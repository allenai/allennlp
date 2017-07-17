from typing import Dict, List, Type

from allennlp.common.checks import ConfigurationError
from allennlp.data import DataIterator, DatasetReader, TokenIndexer, Tokenizer


def _registry_decorator(registry_name: str, registry_dict: Dict[str, Type]):
    """
    There are several different types that we want to create registries for.  They all do the same
    thing, just with a different name and a different dictionary to store the mapping from names to
    classes.  This function allows us to easily define a decorator for any registry we want to
    create.
    """
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


def _get_keys_with_default(registry_dict: Dict[str, Type],
                           registry_name: str,
                           default_name: str) -> List[str]:
    """
    :func:`allennlp.common.Params.pop_choice` allows you to use the first item in a list as a
    default when no choice was provided for a given parameter.  Further, we often get a list of
    options for these choices by calling something like ``Registry.get_dataset_readers()``.  In order
    to have consistent behavior with reasonable defaults, we need the first item in the list of
    ``DatasetReaders`` to be consistent.  We used to achieve that with ``OrderedDicts``, but since
    we switched to a ``Registry`` class, we now specify default keys manually for each registry,
    when a default makes sense.

    This function allows us to specify the default ordering logic once, and re-use it in all of
    the places it's necessary.
    """
    keys = list(registry_dict.keys())
    if default_name not in keys:
        raise ConfigurationError("Default %s (%s) not in list!" % (registry_name, default_name))
    keys.remove(default_name)
    keys.insert(0, default_name)
    return keys


class Registry:
    # Throughout this class, in the `get_*` methods, we have unused import statements.  That is
    # because we use the registry for internal implementations of things, too, and need to be sure
    # they've been imported so they are in the registry by default.

    _dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
    register_dataset_reader = _registry_decorator("dataset reader", _dataset_readers)

    @classmethod
    def get_dataset_readers(cls) -> List[str]:
        import allennlp.data.dataset_readers  # pylint: disable=unused-variable
        return list(cls._dataset_readers.keys())

    @classmethod
    def get_dataset_reader(cls, name: str) -> Type[DatasetReader]:
        import allennlp.data.dataset_readers  # pylint: disable=unused-variable
        return cls._dataset_readers[name]

    _data_iterators = {}  # type: Dict[str, Type[DataIterator]]
    register_data_iterator = _registry_decorator("data iterator", _data_iterators)
    default_data_iterator = "bucket"

    @classmethod
    def get_data_iterators(cls) -> List[str]:
        import allennlp.data.iterators  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._data_iterators, "data iterator", cls.default_data_iterator)

    @classmethod
    def get_data_iterator(cls, name: str) -> Type[DataIterator]:
        import allennlp.data.iterators  # pylint: disable=unused-variable
        return cls._data_iterators[name]

    _tokenizers = {}  # type: Dict[str, Type[Tokenizer]]
    register_tokenizer = _registry_decorator("tokenizer", _tokenizers)
    default_tokenizer = "word"

    @classmethod
    def get_tokenizers(cls) -> List[str]:
        import allennlp.data.tokenizers  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._tokenizers, "tokenizer", cls.default_tokenizer)

    @classmethod
    def get_tokenizer(cls, name: str) -> Type[Tokenizer]:
        import allennlp.data.tokenizers  # pylint: disable=unused-variable
        return cls._tokenizers[name]

    _token_indexers = {}  # type: Dict[str, Type[TokenIndexer]]
    register_token_indexer = _registry_decorator("token indexer", _token_indexers)
    default_token_indexer = "single_id"

    @classmethod
    def get_token_indexers(cls) -> List[str]:
        import allennlp.data.token_indexers  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._token_indexers, "token indexer", cls.default_token_indexer)

    @classmethod
    def get_token_indexer(cls, name: str) -> Type[TokenIndexer]:
        import allennlp.data.token_indexers  # pylint: disable=unused-variable
        return cls._token_indexers[name]
