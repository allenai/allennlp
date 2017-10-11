from collections import defaultdict
from typing import Dict, Generic, List, TypeVar

import numpy

from allennlp.data.vocabulary import Vocabulary

DataArray = TypeVar("DataArray", numpy.ndarray, Dict[str, numpy.ndarray])  # pylint: disable=invalid-name


class Field(Generic[DataArray]):
    """
    A ``Field`` is some piece of a data instance that ends up as an array in a model (either as an
    input or an output).  Data instances are just collections of fields.

    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into data arrays.  The ``Field`` API has methods around both of these steps,
    though they may not be needed for some concrete ``Field`` classes - if your field doesn't have
    any strings that need indexing, you don't need to implement ``count_vocab_items`` or ``index``.
    These methods ``pass`` by default.

    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual arrays.
    """
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.

        A note on this ``counter``: because ``Fields`` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a ``TextField`` from sharing the same ids with labels in a ``LabelField`` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)

        Additionally, a single ``Field`` might want to use multiple namespaces - ``TextFields`` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.

        Because of this, the first key in the ``counter`` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the ``Field`` object, it does not return anything.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like {'num_tokens': 13}.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        numpy array of the correct shape.  This actually returns a list instead of a single array,
        in case there are several related arrays for this field (e.g., a ``TextField`` might have a
        word array and a characters-per-word array).
        """
        raise NotImplementedError

    def empty_field(self) -> 'Field':
        """
        So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
        option ``TextFields``), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_array`, so you don't need to worry about ``get_padding_lengths``,
        ``count_vocab_items``, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in ``TextField``).
        """
        raise NotImplementedError

    @classmethod
    def batch_arrays(cls, array_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        Takes the output of ``Field.as_array()`` from a list of ``Instances`` and merges it into
        one batched array for this ``Field``.  The default implementation here in the base class
        handles cases where ``as_array`` returns a single numpy array per instance, or a dictionary
        of single arrays.  If your subclass returns something other than this, you need to
        override this method.
        """
        if isinstance(array_list[0], dict):
            # This is creating a dict of {token_indexer_key: batch_array} for each
            # token indexer used to index this field. This is mostly utilised by TextFields.
            token_indexer_key_to_batch_dict: Dict[str, List[numpy.ndarray]] = defaultdict(list)
            for encoding_name_dict in array_list:
                for indexer_name, array in encoding_name_dict.items():
                    token_indexer_key_to_batch_dict[indexer_name].append(array)
            return {indexer_name: numpy.asarray(array_list)
                    for indexer_name, array_list in token_indexer_key_to_batch_dict.items()}
        else:
            return numpy.asarray(array_list)
