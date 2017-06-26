from typing import List
from overrides import overrides

from ...common import Params
from .tokenizer import Tokenizer


class CharacterTokenizer(Tokenizer):
    """
    A ``CharacterTokenizer`` splits strings into character tokens.

    Parameters
    ----------
    byte_encoding : str, optional (default=``None``)
        If not ``None``, we will use this encoding to encode the string as bytes, and use the byte
        sequence as characters, instead of the unicode characters in the python string.  E.g., the
        character 'á' would be a single token if this option is ``None``, but it would be two
        tokens if this option is set to ``"utf-8"``: there is one token for the accent and another
        token for the 'a' character.  Note, though, that with a utf-8 encoding, the token for an
        accented 'a' will be different than the token for an unaccented 'a' - hopefully your
        embedding will decide that they are at least similar - while the token for the accent is
        indeed shared across different accented vowels.
    lowercase_characters : ``bool``, optional (default=``False``)
        If ``True``, we will lowercase all of the characters in the text before doing any other
        operation.  You probably do not want to do this, as character vocabularies are generally
        not very large to begin with, but it's an option if you really want it.
    """
    def __init__(self, byte_encoding: str=None, lowercase_characters: bool=False):
        self.byte_encoding = byte_encoding
        self.lowercase_characters = lowercase_characters

    @overrides
    def tokenize(self, text: str) -> List[str]:
        if self.lowercase_characters:
            text = text.lower()
        if self.byte_encoding is not None:
            text = text.encode(self.byte_encoding)
        return list(text)

    @classmethod
    def from_params(cls, params: Params):
        byte_encoding = params.pop('byte_encoding', None)
        lowercase_characters = params.pop('lowercase_characters', False)
        params.assert_empty(cls.__name__)
        return cls(byte_encoding=byte_encoding, lowercase_characters=lowercase_characters)
