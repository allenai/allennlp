"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, Union, Sequence
import codecs
import logging
import os

import tqdm

from allennlp.common.util import namespace_match
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <https://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the ``defaultdict``), and use different
    default values depending on whether the namespace passes the filter.

    To do filtering, we take a sequence of ``non_padded_namespaces``.  This is a list or tuple of
    strings that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_namespaces`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_namespaces : ``Sequence[str]``
        A list or tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that namespace, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """
    def __init__(self,
                 non_padded_namespaces: Sequence[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_namespaces = non_padded_namespaces
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(namespace_match(pattern, key) for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Sequence[str], padding_token: str, oov_token: str) -> None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {padding_token: 0, oov_token: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Sequence[str], padding_token: str, oov_token: str) -> None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {0: padding_token, 1: oov_token},
                                                       lambda: {})


class Vocabulary:
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

    Parameters
    ----------
    counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
        anything.
    min_count : ``int``, optional (default=``1``)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  The default of
        ``1`` means that every word ever seen will be added.
    max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each namespace in the
        ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_namespaces : ``Sequence[str]``, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is ``("*tags", "*labels")``, so as long as your namespace ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.
    """
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: int = 1,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Sequence[str] = DEFAULT_NON_PADDED_NAMESPACES) -> None:
        self._padding_token = DEFAULT_PADDING_TOKEN
        self._oov_token = DEFAULT_OOV_TOKEN
        if not isinstance(max_vocab_size, dict):
            max_vocab_size = defaultdict(lambda: max_vocab_size)  # type: ignore
        self._non_padded_namespaces = non_padded_namespaces
        self._token_to_index = _TokenToIndexDefaultDict(non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(non_padded_namespaces,
                                                        self._padding_token,
                                                        self._oov_token)
        if counter is not None:
            for namespace in counter:
                token_counts = list(counter[namespace].items())
                token_counts.sort(key=lambda x: x[1], reverse=True)
                max_vocab = max_vocab_size.get(namespace)
                if max_vocab:
                    token_counts = token_counts[:max_vocab]
                for token, count in token_counts:
                    if count >= min_count:
                        self.add_token_to_namespace(token, namespace)

    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.

        Parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning("vocabulary serialization directory %s is not empty", directory)

        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        for namespace, mapping in self._index_to_token.items():
            # Each namespace gets written to its own file, in index order.
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    print(mapping[i].replace('\n', '@@NEWLINE@@'), file=token_file)

    @classmethod
    def from_files(cls, directory: str) -> 'Vocabulary':
        """
        Loads a ``Vocabulary`` that was serialized using ``save_to_files``.

        Parameters
        ----------
        directory : ``str``
            The directory containing the serialized vocabulary.
        """
        logger.info("Loading token dictionary from %s.", directory)
        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
            non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

        vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)

        # Check every file in the directory.
        for namespace_filename in os.listdir(directory):
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            namespace = namespace_filename.replace('.txt', '')
            if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                is_padded = False
            else:
                is_padded = True
            filename = os.path.join(directory, namespace_filename)
            vocab.set_from_file(filename, is_padded, namespace=namespace)

        return vocab

    def set_from_file(self,
                      filename: str,
                      is_padded: bool = True,
                      oov_token: str = DEFAULT_OOV_TOKEN,
                      namespace: str = "tokens"):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        Parameters
        ----------
        filename : ``str``
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if ``is_padded``, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : ``bool``, optional (default=True)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            ``True``; while for tag or label vocabularies, this should typically be ``False``.  If
            ``True``, we add a padding token with index 0, and we enforce that the ``oov_token`` is
            present in the file.
        oov_token : ``str``, optional (default=DEFAULT_OOV_TOKEN)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            ``oov_token`` with ``self._oov_token``, because we only use one OOV token across
            namespaces.
        namespace : ``str``, optional (default="tokens")
            What namespace should we overwrite with this vocab file?
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            for i, line in enumerate(input_file.readlines()):
                index = i + 1 if is_padded else i
                token = line[:-1].replace('@@NEWLINE@@', '\n')
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"

    @classmethod
    def from_dataset(cls,
                     dataset,
                     min_count: int = 1,
                     max_vocab_size: Union[int, Dict[str, int]] = None,
                     non_padded_namespaces: Sequence[str] = DEFAULT_NON_PADDED_NAMESPACES) -> 'Vocabulary':
        """
        Constructs a vocabulary given a :class:`.Dataset` and some parameters.  We count all of the
        vocabulary items in the dataset, then pass those counts, and the other parameters, to
        :func:`__init__`.  See that method for a description of what the other parameters do.
        """
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts = defaultdict(lambda: defaultdict(int))  # type: Dict[str, Dict[str, int]]
        for instance in tqdm.tqdm(dataset.instances):
            instance.count_vocab_items(namespace_token_counts)

        return Vocabulary(counter=namespace_token_counts,
                          min_count=min_count,
                          max_vocab_size=max_vocab_size,
                          non_padded_namespaces=non_padded_namespaces)

    @classmethod
    def from_params(cls, params: Params, dataset: "Dataset" = None):
        """
        There are two possible ways to build a vocabulary: using
        :func:`Vocabulary.from_dataset`, or

        Parameters
        ----------
        params: Params, required.
        dataset: Dataset, optional.
            If ``params`` doesn't contain a ``vocabulary_directory`` key,
            the ``Vocabulary`` can be built directly from a ``Dataset``.

        Returns
        -------
        A ``Vocabulary``.
        """
        vocabulary_directory = params.pop("vocabulary_directory", None)
        if not vocabulary_directory and not dataset:
            raise ConfigurationError("You must provide either a Params object containing a "
                                     "vocab directory path or a Dataset to build a vocabulary from.")
        if vocabulary_directory and dataset:
            logger.warn("from_params was passed both a vocabulary_directory and a Dataset. This behaviour "
                        "is undefined, so backing off to building the Vocabulary from the Dataset.")

        if vocabulary_directory and not dataset:
            params.assert_empty("Vocabulary - from files")
            return Vocabulary.from_files(vocabulary_directory)

        return Vocabulary.from_dataset(dataset, **params.as_dict())

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str = 'tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            return self._token_to_index[namespace][self._oov_token]

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int:
        return len(self._token_to_index[namespace])
