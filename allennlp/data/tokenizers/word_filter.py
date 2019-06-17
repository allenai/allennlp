from typing import List
import re
from overrides import overrides

from spacy.lang.en.stop_words import STOP_WORDS

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.common.file_utils import read_set_from_file


class WordFilter(Registrable):
    """
    A ``WordFilter`` removes words from a token list.  Typically, this is for stopword removal,
    though you could feasibly use it for more domain-specific removal if you want.

    Word removal happens `before` stemming, so keep that in mind if you're designing a list of
    words to be removed.
    """
    default_implementation = 'pass_through'

    def filter_words(self, words: List[Token]) -> List[Token]:
        """
        Returns a filtered list of words.
        """
        raise NotImplementedError


@WordFilter.register('pass_through')
class PassThroughWordFilter(WordFilter):
    """
    Does not filter words; it's a no-op.  This is the default word filter.
    """
    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return words


@WordFilter.register('regex')
class RegexFilter(WordFilter):
    """
    A ``RegexFilter`` removes words according to supplied regex patterns.

    Parameters
    ----------
    patterns : ``List[str]``
        Words matching these regex patterns will be removed as stopwords.
    """
    def __init__(self,
                 patterns: List[str]) -> None:
        self._patterns = patterns
        self._joined_pattern = re.compile("|".join(self._patterns))

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        stopwords = [word for word in words
                     if not self._joined_pattern.match(word.text)]
        return stopwords


@WordFilter.register('stopwords')
class StopwordFilter(WordFilter):
    """
    A ``StopwordFilter`` uses a list of stopwords to filter.
    If no file is specified, spaCy's default list of English stopwords is used.
    Words and stopwords are lowercased for comparison.

    Parameters
    ----------
    stopword_file : ``str``, optional
        A filename containing stopwords to filter out (file format is one stopword per line).
    tokens_to_add : ``List[str]``, optional
        A list of tokens to additionally filter out.
    """
    def __init__(self,
                 stopword_file: str = None,
                 tokens_to_add: List[str] = None) -> None:
        self._tokens_to_add = tokens_to_add or []
        if stopword_file is not None:
            self.stopwords = {token.lower() for token in read_set_from_file(stopword_file)}
        else:
            self.stopwords = STOP_WORDS
        for token in self._tokens_to_add:
            self.stopwords.add(token.lower())

    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return [word for word in words if word.text.lower() not in self.stopwords]
