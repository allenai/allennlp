from typing import Callable, Dict, List, Type

import torch
import torch.nn.init

from allennlp.common.checks import ConfigurationError
from allennlp.data import DataIterator, DatasetReader, TokenIndexer, Tokenizer
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, SimilarityFunction, TextFieldEmbedder, TokenEmbedder
from allennlp.training import Regularizer, Model


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

        >>> from allennlp.experiments import Registry
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
    # pylint: disable=too-many-public-methods
    # Throughout this class, in the `get_*` and `list_*` methods, we have unused import statements.
    # That is because we use the registry for internal implementations of things, too, and need to
    # be sure they've been imported so they are in the registry by default.

    #########################
    # Data-related registries
    #########################

    # Dataset Readers

    _dataset_readers = {}  # type: Dict[str, Type[DatasetReader]]
    #: This decorator adds a :class:`DatasetReader` to the registry, with the given name.
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

    # Data Iterators

    _data_iterators = {}  # type: Dict[str, Type[DataIterator]]
    #: This decorator adds a :class:`DataIterator` to the registry, with the given name.
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

    # Tokenizers

    _tokenizers = {}  # type: Dict[str, Type[Tokenizer]]
    #: This decorator adds a :class:`Tokenizer` to the registry, with the given name.
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

    # Token Indexers

    _token_indexers = {}  # type: Dict[str, Type[TokenIndexer]]
    #: This decorator adds a :class:`TokenIndexer` to the registry, with the given name.
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

    #############################
    # Training-related registries
    #############################

    # Regularizers

    _regularizers = {}  # type: Dict[str, Type[Regularizer]]
    #: This decorator adds a :class:`Regularizer` to the registry, with the given name.
    register_regularizer = _registry_decorator("regularizer", _regularizers)
    default_regularizer = "l2"

    @classmethod
    def list_regularizers(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`Regularizer` names.
        """
        import allennlp.training.regularizers  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._regularizers, "regularizer", cls.default_regularizer)

    @classmethod
    def get_regularizer(cls, name) -> Type[Regularizer]:
        """
        Returns the :class:`Regularizer` that has been registered with ``name``.
        """
        import allennlp.training.regularizers  # pylint: disable=unused-variable
        return cls._regularizers[name]

    # Initializers

    _initializers = {
            "normal": torch.nn.init.normal,
            "uniform": torch.nn.init.uniform,
            "orthogonal": torch.nn.init.orthogonal,
            "constant": torch.nn.init.constant,
            "dirac": torch.nn.init.dirac,
            "xavier_normal": torch.nn.init.xavier_normal,
            "xavier_uniform": torch.nn.init.xavier_uniform,
            "kaiming_normal": torch.nn.init.kaiming_normal,
            "kaiming_uniform": torch.nn.init.kaiming_uniform,
            "sparse": torch.nn.init.sparse,
            "eye": torch.nn.init.eye,
    }
    # This decorator adds an ``initializer`` to the registry, with the given name.
    register_initializer = _registry_decorator("initializer", _initializers)
    default_initializer = "normal"

    @classmethod
    def list_initializers(cls) -> List[str]:
        """
        Returns a list of all currently-registered initializer names.
        """
        return _get_keys_with_default(cls._initializers, "initializer", cls.default_initializer)

    @classmethod
    def get_initializer(cls, name) -> Callable[[torch.Tensor], None]:
        """
        Returns the initializer that has been registered with ``name``.
        """
        return cls._initializers[name]

    # Optimizers

    _optimizers = {
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "adadelta": torch.optim.Adadelta,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
    }
    #: This decorator adds an ``optimizer`` to the registry, with the given name.
    register_optimizer = _registry_decorator("optimizer", _optimizers)
    default_optimizer = "adam"

    @classmethod
    def list_optimizers(cls) -> List[str]:
        """
        Returns a list of all currently-registered optimizer names.
        """
        return _get_keys_with_default(cls._optimizers, "optimizer", cls.default_optimizer)

    @classmethod
    def get_optimizer(cls, name) -> torch.optim.Optimizer:
        """
        Returns the optimizer that has been registered with ``name``.
        """
        return cls._optimizers[name]



    ###########################
    # Module-related registries
    ###########################

    # Token Embedders

    _token_embedders = {}  # type: Dict[str, Type[TokenEmbedder]]
    #: This decorator adds a :class:`TokenEmbedder` to the registry, with the given name.
    register_token_embedder = _registry_decorator("token embedder", _token_embedders)
    default_token_embedder = "embedding"

    @classmethod
    def list_token_embedders(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`TokenEmbedder` names.  These take
        a tensor with ids (either single token ids or token character id lists) and return a tensor
        with vectors.
        """
        import allennlp.modules.token_embedders  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._token_embedders, "token embedder",
                                      cls.default_token_embedder)

    @classmethod
    def get_token_embedder(cls, name: str) -> Type[TokenEmbedder]:
        """
        Returns the :clases:`TokenEmbedder` that has been registered with ``name``.  This module
        must take a tensor with ids (either single token ids or token character id lists) and
        return a tensor with vectors.
        """
        import allennlp.modules.token_embedders  # pylint: disable=unused-variable
        return cls._token_embedders[name]

    # Text Field Embedders

    _text_field_embedders = {}  # type: Dict[str, Type[TextFieldEmbedder]]
    #: This decorator adds a :class:`TextFieldEmbedder` to the registry, with the given name.
    register_text_field_embedder = _registry_decorator("text field embedder", _text_field_embedders)
    default_text_field_embedder = "basic"

    @classmethod
    def list_text_field_embedders(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`TextFieldEmbedder` names.  These take the
        dictionary of arrays corresponding to a single ``TextField``, and return a tensor of shape
        ``(batch_size, num_tokens, embedding_dim)``.
        """
        import allennlp.modules.text_field_embedders  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._text_field_embedders, "text field embedder",
                                      cls.default_text_field_embedder)

    @classmethod
    def get_text_field_embedder(cls, name: str) -> Type[TextFieldEmbedder]:
        """
        Returns the :class:`TextFieldEmbedder` that has been registered with ``name``.  This module
        must take the dictionary of arrays corresponding to a single ``TextField``, and return a
        tensor of shape ``(batch_size, num_tokens, embedding_dim)``.
        """
        import allennlp.modules.text_field_embedders  # pylint: disable=unused-variable
        return cls._text_field_embedders[name]

    # Seq2Seq Encoders

    _seq2seq_encoders = {}  # type: Dict[str, Type[Seq2SeqEncoder]]
    #: This decorator adds a :class:`Seq2SeqEncoder` to the registry, with the given name.
    register_seq2seq_encoder = _registry_decorator("seq2seq encoder", _seq2seq_encoders)

    @classmethod
    def list_seq2seq_encoders(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`Seq2SeqEncoder` names.  These take a
        tensor of shape ``(batch_size, sequence_length, input_dim)`` and return a tensor of shape
        ``(batch_size, sequence_length, output_dim)``.
        """
        import allennlp.modules.seq2seq_encoders  # pylint: disable=unused-variable
        return list(cls._seq2seq_encoders.keys())

    @classmethod
    def get_seq2seq_encoder(cls, name: str) -> Type[Seq2SeqEncoder]:
        """
        Returns the :class:`Seq2SeqEncoder` that has been registered with ``name``.  This module
        must take a tensor of shape ``(batch_size, sequence_length, input_dim)`` and return a
        tensor of shape ``(batch_size, sequence_length, output_dim)``.
        """
        import allennlp.modules.seq2seq_encoders  # pylint: disable=unused-variable
        return cls._seq2seq_encoders[name]

    # Seq2Vec Encoders

    _seq2vec_encoders = {}  # type: Dict[str, Type[Seq2VecEncoder]]
    #: This decorator adds a :class:`Seq2VecEncoder` to the registry, with the given name.
    register_seq2vec_encoder = _registry_decorator("seq2vec encoder", _seq2vec_encoders)

    @classmethod
    def list_seq2vec_encoders(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`Seq2VecEncoder` names.  These take a
        tensor of shape ``(batch_size, sequence_length, input_dim)`` and return a tensor of shape
        ``(batch_size, output_dim)``.
        """
        import allennlp.modules.seq2vec_encoders  # pylint: disable=unused-variable
        return list(cls._seq2vec_encoders.keys())

    @classmethod
    def get_seq2vec_encoder(cls, name: str) -> Type[Seq2VecEncoder]:
        """
        Returns the :class:`Seq2VecEncoder` that has been registered with ``name``.  This module
        must take a tensor of shape ``(batch_size, sequence_length, input_dim)`` and return a
        tensor of shape ``(batch_size, output_dim)``.
        """
        import allennlp.modules.seq2vec_encoders  # pylint: disable=unused-variable
        return cls._seq2vec_encoders[name]

    # Models
    _models = {}  # type: Dict[str, Type[Model]]
    #: This decorator adds a :class:`Model` to the registry, with the given name.
    register_model = _registry_decorator("model", _models)

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`Model` names.
        """
        import allennlp.models  # pylint: disable=unused-variable
        return list(cls._models.keys())

    @classmethod
    def get_model(cls, name: str) -> Type[Model]:
        """
        Returns the :class:`Model` that has been registered with ``name``.
        """
        import allennlp.models  # pylint: disable=unused-variable
        return cls._models[name]

    # Similarity functions

    _similarity_functions = {}  # type: Dict[str, Type[SimilarityFunction]]
    #: This decorator adds a :class:`SimilarityFunction` to the registry, with the given name.
    register_similarity_function = _registry_decorator("similarity function", _similarity_functions)
    default_similarity_function = "dot_product"

    @classmethod
    def list_similarity_functions(cls) -> List[str]:
        """
        Returns a list of all currently-registered :class:`SimilarityFunction` names.  These take
        two tensors of the same shape and compute a (possibly parameterized) similarity measure on
        the last dimension.
        """
        import allennlp.modules.similarity_functions  # pylint: disable=unused-variable
        return _get_keys_with_default(cls._similarity_functions, "similarity function",
                                      cls.default_similarity_function)

    @classmethod
    def get_similarity_function(cls, name: str) -> Type[SimilarityFunction]:
        """
        Returns the :class:`SimilarityFunction` that has been registered with ``name``.  This
        module must takes two tensors of the same shape and compute a (possibly parameterized)
        similarity measure on the last dimension.
        """
        import allennlp.modules.similarity_functions  # pylint: disable=unused-variable
        return cls._similarity_functions[name]
