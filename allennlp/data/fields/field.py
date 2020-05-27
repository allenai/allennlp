from copy import deepcopy
from typing import Dict, Generic, List, TypeVar

import torch

from allennlp.data.vocabulary import Vocabulary

DataArray = TypeVar(
    "DataArray", torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]
)


class Field(Generic[DataArray]):
    """
    A `Field` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.

    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into tensors.  The `Field` API has methods around both of these steps, though they
    may not be needed for some concrete `Field` classes - if your field doesn't have any strings
    that need indexing, you don't need to implement `count_vocab_items` or `index`.  These
    methods `pass` by default.

    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual tensors.
    """

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.

        A note on this `counter`: because `Fields` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a `TextField` from sharing the same ids with labels in a `LabelField` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)

        Additionally, a single `Field` might want to use multiple namespaces - `TextFields` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.

        Because of this, the first key in the `counter` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the `Field` object, it does not return anything.

        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like `{'num_tokens': 13}`.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        torch Tensor (or a more complex data structure) of the correct shape.  We also take a
        couple of parameters that are important when constructing torch Tensors.

        # Parameters

        padding_lengths : `Dict[str, int]`
            This dictionary will have the same keys that were produced in
            :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
            relevant dimension, aggregated across all instances in a batch.
        """
        raise NotImplementedError

    def empty_field(self) -> "Field":
        """
        So that `ListField` can pad the number of fields in a list (e.g., the number of answer
        option `TextFields`), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`as_tensor`, so you don't need to worry about `get_padding_lengths`,
        `count_vocab_items`, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in `TextField`).
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        Takes the output of `Field.as_tensor()` from a list of `Instances` and merges it into
        one batched tensor for this `Field`.  The default implementation here in the base class
        handles cases where `as_tensor` returns a single torch tensor per instance.  If your
        subclass returns something other than this, you need to override this method.

        This operation does not modify `self`, but in some cases we need the information
        contained in `self` in order to perform the batching, so this is an instance method, not
        a class method.
        """

        return torch.stack(tensor_list)

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)
