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
        tokens if this option is set to ``"utf-8"``.

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
    """
    def __init__(self,
                 byte_encoding: str = None,
                 lowercase_characters: bool = False,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        if self._lowercase_characters:
            text = text.lower()
        if self._byte_encoding is not None:
            # We add 1 here so that we can still use 0 for masking, no matter what bytes we get out
            # of this.
            tokens = [c + 1 for c in text.encode(self._byte_encoding)]
        else:
            tokens = list(text)  # type: ignore
        for start_token in self._start_tokens:
            tokens.insert(0, start_token)  # type: ignore
        for end_token in self._end_tokens:
            tokens.append(end_token)  # type: ignore
        return tokens, None  # type: ignore

    @classmethod
    def from_params(cls, params: Params) -> 'CharacterTokenizer':
        byte_encoding = params.pop('byte_encoding', None)
        lowercase_characters = params.pop('lowercase_characters', False)
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(byte_encoding=byte_encoding,
                   lowercase_characters=lowercase_characters,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)
