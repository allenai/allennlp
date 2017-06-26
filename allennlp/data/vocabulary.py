from collections import defaultdict
from typing import Dict, List, Union
import codecs
import logging

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _NamespaceDependentDefaultDict(defaultdict):
    """
    Sometimes certain namespaces need padding (like "tokens") and some don't (like
    "labels"), and we want different defaults depending on the namespace.  This class lets us use a
    ``defaultdict`` (https://docs.python.org/2/library/collections.html#collections.defaultdict),
    but have different default values depending on the namespace of the key.

    This class also handles *-namespaces.  In other words, if "*tags" is in non_padded_namespaces
    then "passage_tags", "question_tags", etc. (anything that ends with "tags" will have the
    non_padded default value.
    """
    def __init__(self, non_padded_namespaces: List[str], padded_function, non_padded_function):
        self._non_padded_namespaces = non_padded_namespaces
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        value = None
        for namespace_str in self._non_padded_namespaces:
            if namespace_str[0] == '*' and key.endswith(namespace_str[1:]):
                value = self._non_padded_function()
            elif namespace_str == key:
                value = self._non_padded_function()
        if value is None:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: List[str], padding_token: str, oov_token: str):
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_namespaces,
                                                       lambda: {padding_token: 0, oov_token: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: List[str], padding_token: str, oov_token: str):
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
        ``counter`` can have a separate maximum vocabulariy size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_namespaces : ``List[str]``, optional (default=``["*tags", "*labels"]``)
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        namespaces should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or "*" followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.  As
        long as your namespace ends in "tags" or "labels" (which is true by default for all tag and
        label fields in this code), you don't have to specify anything here.
    """
    def __init__(self,
                 counter: Dict[str, Dict[str, int]]=None,
                 min_count: int=1,
                 max_vocab_size: Union[int, Dict[str, int]]=None,
                 non_padded_namespaces: List[str]=None):
        self._padding_token = "@@PADDING@@"
        self._oov_token = "@@UNKOWN@@"
        if non_padded_namespaces is None:
            non_padded_namespaces = ["*tags", "*labels"]
        if not isinstance(max_vocab_size, dict):
            max_vocab_size = defaultdict(lambda: max_vocab_size)
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

    def set_from_file(self, filename: str, oov_token: str, namespace: str="tokens"):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.

        Parameters
        ----------
        filename : ``str``
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the
            (1-indexed) line number in the file.  Note that this file should contain the OOV token
            string!
        oov_token : ``str``
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.
        namespace : ``str``, optional (default=``"tokens"``)
            What namespace should we overwrite with this vocab file?
        """
        self._oov_token = oov_token
        self._token_to_index[namespace] = {self._padding_token: 0}
        self._index_to_token[namespace] = [self._padding_token]
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            for i, line in enumerate(input_file.readlines()):
                token = line.strip()
                self._token_to_index[namespace][token] = i + 1
                self._index_to_token[namespace].append(token)

    @classmethod
    def from_dataset(cls,
                     dataset,
                     min_count: int=1,
                     max_vocab_size: Union[int, Dict[str, int]]=None,
                     non_padded_namespaces: List[str]=None) -> 'Vocabulary':
        """
        Constructs a vocabulary given a :class:`.Dataset` and some parameters.  We count all of the
        vocabulary items in the dataset, then pass those counts, and the other parameters, to
        :func:`__init__`.  See that method for a description of what the other parameters do.
        """
        logger.info("Fitting token dictionary")
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        for instance in tqdm.tqdm(dataset.instances):
            instance.count_vocab_items(namespace_token_counts)

        return Vocabulary(counter=namespace_token_counts,
                          min_count=min_count,
                          max_vocab_size=max_vocab_size,
                          non_padded_namespaces=non_padded_namespaces)

    def add_token_to_namespace(self, token: str, namespace: str='tokens') -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def get_index_to_token_vocabulary(self, namespace: str='tokens') -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_index(self, token: str, namespace: str='tokens') -> int:
        if token in self._token_to_index[namespace]:
            return self._token_to_index[namespace][token]
        else:
            return self._token_to_index[namespace][self._oov_token]

    def get_token_from_index(self, index: int, namespace: str='tokens') -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str='tokens') -> int:
        return len(self._token_to_index[namespace])
