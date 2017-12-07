from typing import Dict, List, TypeVar, Generic

from allennlp.common import Params, Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary

TokenType = TypeVar("TokenType", int, List[int])  # pylint: disable=invalid-name

class TokenIndexer(Generic[TokenType], Registrable):
    """
    A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.
    This class both converts strings into numerical values, with the help of a
    :class:`~allennlp.data.vocabulary.Vocabulary`, and it produces actual arrays.

    Tokens can be represented as single IDs (e.g., the word "cat" gets represented by the number
    34), or as lists of character IDs (e.g., "cat" gets represented by the numbers [23, 10, 18]),
    or in some other way that you can come up with (e.g., if you have some structured input you
    want to represent in a special way in your data arrays, you can do that here).
    """
    default_implementation = 'single_id'

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        """
        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
        data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
        token).  This method takes a token and a dictionary of counts and increments counts for
        whatever vocabulary items are present in the token.  If this is a single token ID
        representation, the vocabulary item is likely the token itself.  If this is a token
        characters representation, the vocabulary items are all of the characters in the token.
        """
        raise NotImplementedError

    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> TokenType:
        """
        Takes a string token and converts it into indices.  This could return an ID for the token
        from the vocabulary, or it could split the token into characters and return a list of
        IDs for each character from the vocabulary, or something else.
        """
        raise NotImplementedError

    def get_padding_token(self) -> TokenType:
        """
        When we need to add padding tokens, what should they look like?  This method returns a
        "blank" token of whatever type is returned by :func:`token_to_indices`.
        """
        raise NotImplementedError

    def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
        """
        This method returns a padding dictionary for the given token that specifies lengths for
        all arrays that need padding.  For example, for single ID tokens the returned dictionary
        will be empty, but for a token characters representation, this will return the number
        of characters in the token.
        """
        raise NotImplementedError

    def pad_token_sequence(self,
                           tokens: List[TokenType],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[TokenType]:
        """
        This method pads a list of tokens to ``desired_num_tokens`` and returns a padded copy of the
        input tokens.  If the input token list is longer than ``desired_num_tokens`` then it will be
        truncated.

        ``padding_lengths`` is used to provide supplemental padding parameters which are needed
        in some cases.  For example, it contains the widths to pad characters to when doing
        character-level padding.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'TokenIndexer':  # type: ignore
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        return cls.by_name(choice).from_params(params)

    @classmethod
    def dict_from_params(cls, params: Params) -> 'Dict[str, TokenIndexer]':  # type: ignore
        """
        We typically use ``TokenIndexers`` in a dictionary, with each ``TokenIndexer`` getting a
        name.  The specification for this in a ``Params`` object is typically ``{"name" ->
        {indexer_params}}``.  This method reads that whole set of parameters and returns a
        dictionary suitable for use in a ``TextField``.

        Because default values for token indexers are typically handled in the calling class to
        this and are based on checking for ``None``, if there were no parameters specifying any
        token indexers in the given ``params``, we return ``None`` instead of an empty dictionary.
        """
        token_indexers = {}
        for name, indexer_params in params.items():
            token_indexers[name] = cls.from_params(indexer_params)
        if token_indexers == {}:
            token_indexers = None
        return token_indexers
