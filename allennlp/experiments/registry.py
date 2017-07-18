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
    options for these choices by calling something like ``Registry.list_dataset_readers()``.  In
    order to have consistent behavior with reasonable defaults, we need the first item in the list
    of ``DatasetReaders`` to be consistent.  We used to achieve that with ``OrderedDicts``, but
    since we switched to a ``Registry`` class, we now specify default keys manually for each
    registry, when a default makes sense.

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
    """
    The ``Registry`` is where we store mappings from names that appear in JSON configuration files
    to classes that we can instantiate.  This class underpins the whole ``from_params()`` paradigm
    of running experiments.

    If you want to use a particular :class:`DataIterator` when running an experiment, for example,
    you might specify something like this in a configuration file::

        "data_iterator": {
            "type": "bucket",
            "batch_size": 64,
            ...
        }

    This class is where the mapping from the type name "bucket" to the class
    :class:`~allennlp.data.iterators.BucketIterator` is made.

    It would be easy for us to just hard-code some mappings from names that we pick to classes that
    we've implemented, but that would make it hard for you to write your own implementations of
    these types and extend the library.  Instead, we use the ``Registry``, so that you can register
    names to be included in this mapping and use our experiment framework with classes that you've
    implemented.  There are decorators that you can use to add classes of various types to the
    registry, such as :func:`register_data_iterator()`.  You use these decorators as follows::

        from allennlp.experiments import Registry
        @Registry.register_data_iterator("my_fancy_name")
        class MyFancyDataIterator(DataIterator):
            ...

    We use the registry for all of our classes in this library, too, so you can look at our code to
    see how the registry decorators are used.

    To see all of the options available for a particular type, you can use the
    ``Registry.list_*()`` methods::

        >>> from allennlp.experiments import Regsitry
        >>> Registry.list_data_iterators()
        ["bucket", "basic", "adaptive"]  # the default options show up here (this list may be out-dated)

    If you've implemented a new ``DataIterator`` (or some other type that's managed by the
    ``Registry``), and registered it with the proper decorator, all you need to do to make sure you
    can use the experiment framework properly is import the class::

        >>> from allennlp.experiments import Registry
        >>> from my_library import MyFancyDataIterator  # as defined in example code block above
        >>> Registry.list_data_iterators()
        ["bucket", "basic", "adaptive", "my_fancy_name"]

    One final point: for some types there is a reasonable default to use.  For tokenizers, for
    instance, most models are probably fine using the default
    :class:`~allennlp.data.tokenizers.WordTokenizer`.  In these cases, the ``Registry`` specifies a
    default value, which will get used if a key is omitted from parameters.  When creating a
    :class:`DatasetReader` for the SNLI dataset, for instance, you might have parameters that look
    like this::

        "dataset_reader": {
            "type": "snli",
            "tokenizer": {
                "type": "word"
            }
            ...
        }

    In this case, because using the word tokenizer is the default (instead of tokenizing by
    characters, or bytes, or something else), we can omit the entire key, giving a simpler
    parameter file::

        "dataset_reader": {
            "type": "snli",
            ...
        }

    When using default parameter values this way, the default values that were `actually used`
    still get logged with a special ``PARAM`` logging level, so you can still recover a full
    configuration just from examining the log file, even if the default value changes over time.
    """
    # Throughout this class, in the `get_*` methods, we have unused import statements.  That is
    # because we use the registry for internal implementations of things, too, and need to be sure
    # they've been imported so they are in the registry by default.

    _dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
    #: This decorator adds a :class:`DatasetReader` to the regsitry, with the given name.
    register_dataset_reader = _registry_decorator("dataset reader", _dataset_readers)

    @classmethod
    def list_dataset_readers(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`DatasetReader` names.
        """
        import allennlp.data.dataset_readers  # pylint: disable=unused-variable
        return list(cls._dataset_readers.keys())

    @classmethod
    def get_dataset_reader(cls, name: str) -> Type[DatasetReader]:
        """
        Returns the :class:`DatasetReader` that has been registered with ``name``.
        """
        import allennlp.data.dataset_readers  # pylint: disable=unused-variable
        return cls._dataset_readers[name]

    _data_iterators = {}  # type: Dict[str, Type[DataIterator]]
    #: This decorator adds a :class:`DataIterator` to the regsitry, with the given name.
    register_data_iterator = _registry_decorator("data iterator", _data_iterators)
    default_data_iterator = "bucket"

    @classmethod
    def list_data_iterators(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`DataIterator` names.
        """
        import allennlp.data.iterators  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._data_iterators, "data iterator", cls.default_data_iterator)

    @classmethod
    def get_data_iterator(cls, name: str) -> Type[DataIterator]:
        """
        Returns the :class:`DataIterator` that has been registered with ``name``.
        """
        import allennlp.data.iterators  # pylint: disable=unused-variable
        return cls._data_iterators[name]

    _tokenizers = {}  # type: Dict[str, Type[Tokenizer]]
    #: This decorator adds a :class:`Tokenizer` to the regsitry, with the given name.
    register_tokenizer = _registry_decorator("tokenizer", _tokenizers)
    default_tokenizer = "word"

    @classmethod
    def list_tokenizers(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`Tokenizer` names.
        """
        import allennlp.data.tokenizers  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._tokenizers, "tokenizer", cls.default_tokenizer)

    @classmethod
    def get_tokenizer(cls, name: str) -> Type[Tokenizer]:
        """
        Returns the :class:`Tokenizer` that has been registered with ``name``.
        """
        import allennlp.data.tokenizers  # pylint: disable=unused-variable
        return cls._tokenizers[name]

    _token_indexers = {}  # type: Dict[str, Type[TokenIndexer]]
    #: This decorator adds a :class:`TokenIndexer` to the regsitry, with the given name.
    register_token_indexer = _registry_decorator("token indexer", _token_indexers)
    default_token_indexer = "single_id"

    @classmethod
    def list_token_indexers(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`TokenIndexer` names.
        """
        import allennlp.data.token_indexers  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._token_indexers, "token indexer", cls.default_token_indexer)

    @classmethod
    def get_token_indexer(cls, name: str) -> Type[TokenIndexer]:
        """
        Returns the :class:`TokenIndexer` that has been registered with ``name``.
        """
        import allennlp.data.token_indexers  # pylint: disable=unused-variable
        return cls._token_indexers[name]
