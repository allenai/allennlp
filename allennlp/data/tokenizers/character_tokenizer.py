from typing import List, Union

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("character")
class CharacterTokenizer(Tokenizer):
    """
    A `CharacterTokenizer` splits strings into character tokens.

    Registered as a `Tokenizer` with name "character".

    # Parameters

    byte_encoding : `str`, optional (default=`None`)
        If not `None`, we will use this encoding to encode the string as bytes, and use the byte
        sequence as characters, instead of the unicode characters in the python string.  E.g., the
        character 'á' would be a single token if this option is `None`, but it would be two
        tokens if this option is set to `"utf-8"`.

        If this is not `None`, `tokenize` will return a `List[int]` instead of a
        `List[str]`, and we will bypass the vocabulary in the `TokenIndexer`.

    lowercase_characters : `bool`, optional (default=`False`)
        If `True`, we will lowercase all of the characters in the text before doing any other
        operation.  You probably do not want to do this, as character vocabularies are generally
        not very large to begin with, but it's an option if you really want it.

    start_tokens : `List[str]`, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  If
        using byte encoding, this should actually be a `List[int]`, not a `List[str]`.

    end_tokens : `List[str]`, optional
        If given, these tokens will be added to the end of every string we tokenize.  If using byte
        encoding, this should actually be a `List[int]`, not a `List[str]`.

    """

    def __init__(
        self,
        byte_encoding: str = None,
        lowercase_characters: bool = False,
        start_tokens: List[Union[str, int]] = None,
        end_tokens: List[Union[str, int]] = None,
    ) -> None:
        # TODO(brendanr): Add length truncation.
        self._byte_encoding = byte_encoding
        self._lowercase_characters = lowercase_characters
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        if self._lowercase_characters:
            text = text.lower()
        if self._byte_encoding is not None:
            # We add 1 here so that we can still use 0 for masking, no matter what bytes we get out
            # of this.
            tokens = [Token(text_id=c + 1) for c in text.encode(self._byte_encoding)]
        else:
            tokens = [Token(t) for t in list(text)]
        for start_token in self._start_tokens:
            if isinstance(start_token, int):
                token = Token(text_id=start_token, idx=0)
            else:
                token = Token(text=start_token, idx=0)
            tokens.insert(0, token)
        for end_token in self._end_tokens:
            if isinstance(end_token, int):
                token = Token(text_id=end_token, idx=0)
            else:
                token = Token(text=end_token, idx=0)
            tokens.append(token)
        return tokens

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
