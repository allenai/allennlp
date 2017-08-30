from typing import List, Tuple

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("character")
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

        If this is not ``None``, ``tokenize`` will return a ``List[int]`` instead of a
        ``List[str]``, and we will bypass the vocabulary in the ``TokenIndexer``.
    lowercase_characters : ``bool``, optional (default=``False``)
        If ``True``, we will lowercase all of the characters in the text before doing any other
        operation.  You probably do not want to do this, as character vocabularies are generally
        not very large to begin with, but it's an option if you really want it.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  If
        using byte encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.  If using byte
        encoding, this should actually be a ``List[int]``, not a ``List[str]``.
    padding_index : ``int``, optional
        If you're using byte encoding, we're bypassing the dictionary, and 0 might be a valid byte
        for some inputs.  If you need to set the padding token to something other than 0, you can
        do so here.  If this parameter is omitted, we will pad with zeros.
    """
    def __init__(self,
                 byte_encoding: str = None,
                 lowercase_characters: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 padding_index: int = None) -> None:
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []
        self.padding_index = padding_index

    @overrides
    def tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if self._lowercase_characters:
            text = text.lower()
        if self._byte_encoding is not None:
            return list(text.encode(self._byte_encoding)), None  # type: ignore
        return list(text), None

    @classmethod
    def from_params(cls, params: Params) -> 'CharacterTokenizer':
        byte_encoding = params.pop('byte_encoding', None)
        lowercase_characters = params.pop('lowercase_characters', False)
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        padding_index = params.pop('padding_index', None)
        params.assert_empty(cls.__name__)
        return cls(byte_encoding=byte_encoding,
                   lowercase_characters=lowercase_characters,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens,
                   padding_index=padding_index)
