from typing import Dict, List, TypeVar, Generic

from allennlp.common import Registrable
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

    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[TokenType]]:
        """
        Takes a list of tokens and converts them to one or more sets of indices.
        This could be just an ID for each token from the vocabulary.
        Or it could split each token into characters and return one ID per character.
        Or (for instance, in the case of byte-pair encoding) there might not be a clean
        mapping from individual tokens to indices.
        """
        raise NotImplementedError

    def get_padding_token(self) -> TokenType:
        """
        When we need to add padding tokens, what should they look like?  This method returns a
        "blank" token of whatever type is returned by :func:`tokens_to_indices`.
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
                           tokens: Dict[str, List[TokenType]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[TokenType]]:
        """
        This method pads a list of tokens to ``desired_num_tokens`` and returns a padded copy of the
        input tokens.  If the input token list is longer than ``desired_num_tokens`` then it will be
        truncated.

        ``padding_lengths`` is used to provide supplemental padding parameters which are needed
        in some cases.  For example, it contains the widths to pad characters to when doing
        character-level padding.
        """
        raise NotImplementedError

    def get_keys(self, index_name: str) -> List[str]:
        """
        Return a list of the keys this indexer return from ``tokens_to_indices``.
        """
        # pylint: disable=no-self-use
        return [index_name]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
