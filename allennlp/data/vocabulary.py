"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

import codecs
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING

from filelock import FileLock

from allennlp.common.util import namespace_match
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm

if TYPE_CHECKING:
    from allennlp.data import instance as adi  # noqa


logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"
_NEW_LINE_REGEX = re.compile(r"\n|\r\n")


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a [defaultdict]
    (https://docs.python.org/2/library/collections.html#collections.defaultdict) where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the `defaultdict`), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a set of `non_padded_namespaces`.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with `*`.  In other words, if `*tags` is in `non_padded_namespaces` then
    `passage_tags`, `question_tags`, etc. (anything that ends with `tags`) will have the
    `non_padded` default value.

    # Parameters

    non_padded_namespaces : `Iterable[str]`
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use `non_padded_function` to initialize the value for that namespace, and
        we will use `padded_function` otherwise.
    padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], Any],
        non_padded_function: Callable[[], Any],
    ) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super().__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {}
        )


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(
            non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {}
        )


def _read_pretrained_tokens(embeddings_file_uri: str) -> List[str]:
    # Moving this import to the top breaks everything (cycling import, I guess)
    from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile

    logger.info("Reading pretrained tokens from: %s", embeddings_file_uri)
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file_uri) as embeddings_file:
        for line_number, line in enumerate(Tqdm.tqdm(embeddings_file), start=1):
            token_end = line.find(" ")
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + "..." if len(line) > 20 else line
                logger.warning("Skipping line number %d: %s", line_number, line_begin)
    return tokens


class Vocabulary(Registrable):
    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different namespaces, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a namespace; by default we use the 'tokens'
    namespace, and you can omit the namespace argument everywhere and just use the default.

    This class is registered as a `Vocabulary` with four different names, which all point to
    different `@classmethod` constructors found in this class.  `from_instances` is registered as
    "from_instances", `from_files` is registered as "from_files", `from_files_and_instances` is
    registered as "extend", and `empty` is registered as "empty".  If you are using a configuration
    file to construct a vocabulary, you can use any of those strings as the "type" key in the
    configuration file to use the corresponding `@classmethod` to construct the object.
    "from_instances" is the default.  Look at the docstring for the `@classmethod` to see what keys
    are allowed in the configuration file (when there is an `instances` argument to the
    `@classmethod`, it will be passed in separately and does not need a corresponding key in the
    configuration file).

    # Parameters

    counter : `Dict[str, Dict[str, int]]`, optional (default=`None`)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is `None`, we just won't initialize the vocabulary with
        anything.

    min_count : `Dict[str, int]`, optional (default=`None`)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `namespace-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a namespace does not have a key in the given dictionary, we
        will add all seen tokens to that namespace.

    max_vocab_size : `Union[int, Dict[str, int]]`, optional (default=`None`)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        `counter` can have a separate maximum vocabulary size.  Any missing key will have a value
        of `None`, which means no cap on the vocabulary size.

    non_padded_namespaces : `Iterable[str]`, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or `*` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is `("*tags", "*labels")`, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.

    pretrained_files : `Dict[str, str]`, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        namespace. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of `only_include_pretrained_words`.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.

    min_pretrained_embeddings : `Dict[str, int]`, optional
        If provided, specifies for each namespace a minimum number of lines (typically the
        most common words) to keep from pretrained embedding files, even for words not
        appearing in the data.

    only_include_pretrained_words : `bool`, optional (default=`False`)
        This defines the strategy for using any pretrained embedding files which may have been
        specified in `pretrained_files`. If False, an inclusive strategy is used: and words
        which are in the `counter` and in the pretrained file are added to the `Vocabulary`,
        regardless of whether their count exceeds `min_count` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least `min_count`).

    tokens_to_add : `Dict[str, List[str]]`, optional (default=`None`)
        If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.

    padding_token : `str`,  optional (default=`DEFAULT_PADDING_TOKEN`)
        If given, this the string used for padding.

    oov_token : `str`,  optional (default=`DEFAULT_OOV_TOKEN`)
        If given, this the string used for the out of vocabulary (OOVs) tokens.

    """

    default_implementation = "from_instances"

    def __init__(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> None:
        self._padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        self._oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        self._non_padded_namespaces = set(non_padded_namespaces)

        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )

        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None

        # Made an empty vocabulary, now extend it.
        self._extend(
            counter,
            min_count,
            max_vocab_size,
            non_padded_namespaces,
            pretrained_files,
            only_include_pretrained_words,
            tokens_to_add,
            min_pretrained_embeddings,
        )

    @classmethod
    def from_instances(
        cls,
        instances: Iterable["adi.Instance"],
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.

        The `instances` parameter does not get an entry in a typical AllenNLP configuration file,
        but the other parameters do (if you want non-default parameters).
        """
        logger.info("Fitting token dictionary from dataset.")
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)

        return cls(
            counter=namespace_token_counts,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token,
        )

    @classmethod
    def from_files(
        cls,
        directory: str,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Loads a `Vocabulary` that was serialized using `save_to_files`.

        # Parameters

        directory : `str`
            The directory containing the serialized vocabulary.
        """
        logger.info("Loading token dictionary from %s.", directory)
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        # We use a lock file to avoid race conditions where multiple processes
        # might be reading/writing from/to the same vocab files at once.
        with FileLock(os.path.join(directory, ".lock")):
            with codecs.open(
                os.path.join(directory, NAMESPACE_PADDING_FILE), "r", "utf-8"
            ) as namespace_file:
                non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

            vocab = cls(
                non_padded_namespaces=non_padded_namespaces,
                padding_token=padding_token,
                oov_token=oov_token,
            )

            # Check every file in the directory.
            for namespace_filename in os.listdir(directory):
                if namespace_filename == NAMESPACE_PADDING_FILE:
                    continue
                if namespace_filename.startswith("."):
                    continue
                namespace = namespace_filename.replace(".txt", "")
                if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                    is_padded = False
                else:
                    is_padded = True
                filename = os.path.join(directory, namespace_filename)
                vocab.set_from_file(filename, is_padded, namespace=namespace, oov_token=oov_token)

        return vocab

    @classmethod
    def from_files_and_instances(
        cls,
        instances: Iterable["adi.Instance"],
        directory: str,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ) -> "Vocabulary":
        """
        Extends an already generated vocabulary using a collection of instances.

        The `instances` parameter does not get an entry in a typical AllenNLP configuration file,
        but the other parameters do (if you want non-default parameters).  See `__init__` for a
        description of what the other parameters mean.
        """
        vocab = cls.from_files(directory, padding_token, oov_token)
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        vocab._extend(
            counter=namespace_token_counts,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
        )
        return vocab

    @classmethod
    def empty(cls) -> "Vocabulary":
        """
        This method returns a bare vocabulary instantiated with `cls()` (so, `Vocabulary()` if you
        haven't made a subclass of this object).  The only reason to call `Vocabulary.empty()`
        instead of `Vocabulary()` is if you are instantiating this object from a config file.  We
        register this constructor with the key "empty", so if you know that you don't need to
        compute a vocabulary (either because you're loading a pre-trained model from an archive
        file, you're using a pre-trained transformer that has its own vocabulary, or something
        else), you can use this to avoid having the default vocabulary construction code iterate
        through the data.
        """
        return cls()

    def set_from_file(
        self,
        filename: str,
        is_padded: bool = True,
        oov_token: str = DEFAULT_OOV_TOKEN,
        namespace: str = "tokens",
    ):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        # Parameters

        filename : `str`
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if `is_padded`, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : `bool`, optional (default=`True`)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            `True`; while for tag or label vocabularies, this should typically be `False`.  If
            `True`, we add a padding token with index 0, and we enforce that the `oov_token` is
            present in the file.
        oov_token : `str`, optional (default=`DEFAULT_OOV_TOKEN`)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            `oov_token` with `self._oov_token`, because we only use one OOV token across
            namespaces.
        namespace : `str`, optional (default=`"tokens"`)
            What namespace should we overwrite with this vocab file?
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, "r", "utf-8") as input_file:
            lines = _NEW_LINE_REGEX.split(input_file.read())
            # Be flexible about having final newline or not
            if lines and lines[-1] == "":
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index = i + 1 if is_padded else i
                token = line.replace("@@NEWLINE@@", "\n")
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"

    def extend_from_instances(self, instances: Iterable["adi.Instance"]) -> None:
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)
        self._extend(counter=namespace_token_counts)

    def extend_from_vocab(self, vocab: "Vocabulary") -> None:
        """
        Adds all vocabulary items from all namespaces in the given vocabulary to this vocabulary.
        Useful if you want to load a model and extends its vocabulary from new instances.

        We also add all non-padded namespaces from the given vocabulary to this vocabulary.
        """
        self._non_padded_namespaces.update(vocab._non_padded_namespaces)
        self._token_to_index._non_padded_namespaces.update(vocab._non_padded_namespaces)
        self._index_to_token._non_padded_namespaces.update(vocab._non_padded_namespaces)
        for namespace in vocab.get_namespaces():
            for token in vocab.get_token_to_index_vocabulary(namespace):
                self.add_token_to_namespace(token, namespace)

    def _extend(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
    ) -> None:
        """
        This method can be used for extending already generated vocabulary.  It takes same
        parameters as Vocabulary initializer. The `_token_to_index` and `_index_to_token`
        mappings of calling vocabulary will be retained.  It is an inplace operation so None will be
        returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        # Make sure vocabulary extension is safe.
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}

        for namespace in current_namespaces & extension_namespaces:
            # if new namespace was already present
            # Either both should be padded or none should be.
            original_padded = not any(
                namespace_match(pattern, namespace) for pattern in self._non_padded_namespaces
            )
            extension_padded = not any(
                namespace_match(pattern, namespace) for pattern in non_padded_namespaces
            )
            if original_padded != extension_padded:
                raise ConfigurationError(
                    "Common namespace {} has conflicting ".format(namespace)
                    + "setting of padded = True/False. "
                    + "Hence extension cannot be done."
                )

        # Add new non-padded namespaces for extension
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:
            if namespace in pretrained_files:
                pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0:
                    tokens_old = tokens_to_add.get(namespace, [])
                    tokens_new = pretrained_list[:min_embeddings]
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            else:
                pretrained_set = None
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[namespace]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set and count >= min_count.get(namespace, 1):
                            self.add_token_to_namespace(token, namespace)
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def __getstate__(self):
        """
        Need to sanitize defaultdict and defaultdict-like objects
        by converting them to vanilla dicts when we pickle the vocabulary.
        """
        state = copy.copy(self.__dict__)
        state["_token_to_index"] = dict(state["_token_to_index"])
        state["_index_to_token"] = dict(state["_index_to_token"])

        if "_retained_counter" in state:
            state["_retained_counter"] = {
                key: dict(value) for key, value in state["_retained_counter"].items()
            }

        return state

    def __setstate__(self, state):
        """
        Conversely, when we unpickle, we need to reload the plain dicts
        into our special DefaultDict subclasses.
        """

        self.__dict__ = copy.copy(state)
        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._token_to_index.update(state["_token_to_index"])
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token.update(state["_index_to_token"])

    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.

        # Parameters

        directory : `str`
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning("vocabulary serialization directory %s is not empty", directory)

        # We use a lock file to avoid race conditions where multiple processes
        # might be reading/writing from/to the same vocab files at once.
        with FileLock(os.path.join(directory, ".lock")):
            with codecs.open(
                os.path.join(directory, NAMESPACE_PADDING_FILE), "w", "utf-8"
            ) as namespace_file:
                for namespace_str in self._non_padded_namespaces:
                    print(namespace_str, file=namespace_file)

            for namespace, mapping in self._index_to_token.items():
                # Each namespace gets written to its own file, in index order.
                with codecs.open(
                    os.path.join(directory, namespace + ".txt"), "w", "utf-8"
                ) as token_file:
                    num_tokens = len(mapping)
                    start_index = 1 if mapping[0] == self._padding_token else 0
                    for i in range(start_index, num_tokens):
                        print(mapping[i].replace("\n", "@@NEWLINE@@"), file=token_file)

    def is_padded(self, namespace: str) -> bool:
        """
        Returns whether or not there are padding and OOV tokens added to the given namespace.
        """
        return self._index_to_token[namespace][0] == self._padding_token

    def add_token_to_namespace(self, token: str, namespace: str = "tokens") -> int:
        """
        Adds `token` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings, or saving and loading will break."
                "  Got %s (with type %s)" % (repr(token), type(token))
            )
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = "tokens") -> List[int]:
        """
        Adds `tokens` to the index, if they are not already present.  Either way, we return the
        indices of the tokens in the order that they were given.
        """
        return [self.add_token_to_namespace(token, namespace) for token in tokens]

    def get_index_to_token_vocabulary(self, namespace: str = "tokens") -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = "tokens") -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        try:
            return self._token_to_index[namespace][token]
        except KeyError:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error("Namespace: %s", namespace)
                logger.error("Token: %s", token)
                raise KeyError(
                    f"'{token}' not found in vocab namespace '{namespace}', and namespace "
                    f"does not contain the default OOV token ('{self._oov_token}')"
                )

    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        return len(self._token_to_index[namespace])

    def get_namespaces(self) -> Set[str]:
        return set(self._index_to_token.keys())

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = "Vocabulary with namespaces:\n"
        non_padded_namespaces = f"\tNon Padded Namespaces: {self._non_padded_namespaces}\n"
        namespaces = [
            f"\tNamespace: {name}, Size: {self.get_vocab_size(name)} \n"
            for name in self._index_to_token
        ]
        return " ".join([base_string, non_padded_namespaces] + namespaces)

    def __repr__(self) -> str:
        # This is essentially the same as __str__, but with no newlines
        base_string = "Vocabulary with namespaces: "
        namespaces = [
            f"{name}, Size: {self.get_vocab_size(name)} ||" for name in self._index_to_token
        ]
        non_padded_namespaces = f"Non Padded Namespaces: {self._non_padded_namespaces}"
        return " ".join([base_string] + namespaces + [non_padded_namespaces])

    def print_statistics(self) -> None:
        if self._retained_counter:
            logger.info(
                "Printed vocabulary statistics are only for the part of the vocabulary generated "
                "from instances. If vocabulary is constructed by extending saved vocabulary with "
                "dataset instances, the directly loaded portion won't be considered here."
            )
            print("\n\n----Vocabulary Statistics----\n")
            # Since we don't saved counter info, it is impossible to consider pre-saved portion.
            for namespace in self._retained_counter:
                tokens_with_counts = list(self._retained_counter[namespace].items())
                tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
                print(f"\nTop 10 most frequent tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tFrequency: {freq}")
                # Now sort by token length, not frequency
                tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)

                print(f"\nTop 10 longest tokens in namespace '{namespace}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")

                print(f"\nTop 10 shortest tokens in namespace '{namespace}':")
                for token, freq in reversed(tokens_with_counts[-10:]):
                    print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")
        else:
            # _retained_counter would be set only if instances were used for vocabulary construction.
            logger.info(
                "Vocabulary statistics cannot be printed since "
                "dataset instances were not used for its construction."
            )


# We can't decorate `Vocabulary` with `Vocabulary.register()`, because `Vocabulary` hasn't been
# defined yet.  So we put these down here.
Vocabulary.register("from_instances", constructor="from_instances")(Vocabulary)
Vocabulary.register("from_files", constructor="from_files")(Vocabulary)
Vocabulary.register("extend", constructor="from_files_and_instances")(Vocabulary)
Vocabulary.register("empty", constructor="empty")(Vocabulary)
