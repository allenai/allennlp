from typing import Dict, List, TypeVar, Generic

from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params, Registrable

TokenType = TypeVar("TokenType", int, List[int])  # pylint: disable=invalid-name

class TokenIndexer(Generic[TokenType], Registrable):
    """
    A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.
    This class both converts strings into numerical values, with the help of a
    :class:`~allennlp.data.vocabulary.Vocabulary`,
    and it produces actual arrays.

    Tokens can be represented as single IDs (e.g., the word "cat" gets represented by the number
    34), or as lists of character IDs (e.g., "cat" gets represented by the numbers [23, 10, 18]),
    or in some other way that you can come up with (e.g., if you have some structured input you
    want to represent in a special way in your data arrays, you can do that here).
    """
    default_implementation = 'single_id'

    def count_vocab_items(self, token: str, counter: Dict[str, Dict[str, int]]):
        """
        The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
        data (possibly doing some frequency filtering and using an OOV token).  This method takes
        a token and a dictionary of counts and increments counts for whatever vocabulary items are
        present in the token.  If this is a single token ID representation, the vocabulary item is
        likely the token itself.  If this is a token characters representation, the vocabulary
        items are all of the characters in the token.
        """
        raise NotImplementedError

    def token_to_indices(self, token: str, vocabulary: Vocabulary) -> TokenType:
        """
        Takes a string token and converts it into indices in some fashion.  This could be returning
        an ID for the token from the vocabulary, or it could be splitting the token into characters
        and return a list of IDs for each character from the vocabulary, or something else.
        """
        raise NotImplementedError

    def get_input_shape(self, num_tokens: int, padding_lengths: Dict[str, int]):
        """
        Returns the shape of an input array containing tokens representated by this
        ``TokenIndexer``, not including the batch size.  ``padding_lengths`` contains the
        same keys returned by :func:`get_padding_lengths`.  For single ID tokens, this shape will
        just be ``(num_tokens,)``; it will be more complicated for more complex representations.
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
        This method returns a padding dictionary for the given token.  For single ID tokens, e.g.,
        this dictionary will be empty, but for a token characters representation, this will return
        the number of characters in the token.
        """
        raise NotImplementedError

    def pad_token_sequence(self,
                           tokens: List[TokenType],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[TokenType]:
        """
        This method pads a list of tokens to ``desired_num_tokens``, including any necessary
        internal padding using whatever lengths are relevant in ``padding_lengths``, returning a
        padded copy of the input list.  If each token is a single ID, this just adds 0 to the
        sequence (or truncates the sequence, if necessary).  If each token is, e.g., a list of
        characters, this method will pad both the characters and the number of tokens.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'TokenIndexer':  # type: ignore
        choice = params.pop_choice('type', cls.list_available(), default_to_first_choice=True)
        return cls.by_name(choice).from_params(params)
